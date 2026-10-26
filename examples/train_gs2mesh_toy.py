#
# Peizhi Yan
# Copyright (C) 2026
#

"""
Gaussians -> mesh -> mesh + fewer Gaussians: a toy pipeline that ends with a
hybrid model that is better than the Gaussians it started from, with fewer
parameters.

Ground truth: the detailed textured Spot mesh (assets/data/spot), lit by a point
light, pre-rendered (color and silhouette) from 24 orbit views (training) and 6
other views (testing). No ground-truth geometry is used for training.

  stage 1 (Gaussians):  N Gaussians, randomly initialized in a box around the
                        scene, trained on the images (plain 3DGS)
  stage 2 (extraction): depth and silhouette of the trained Gaussians, rendered
                        from many views, are fused into a truncated signed
                        distance volume; marching cubes gives a mesh (largest
                        connected component), with vertex colors from the
                        nearest Gaussians (with the light's diffuse shading divided out)
  stage 3 (mesh):       the mesh vertices (large steps parameterization) and its
                        vertex colors are trained on the images
  stage 4 (hybrid):     N / 4 Gaussians are added on the mesh, and the mesh and
                        the Gaussians are trained together through Gmesh

Baselines: the stage 1 Gaussians trained for as many more iterations as stages
3 + 4, and N / 4 Gaussians alone (random initialization, as many iterations as
all stages). Every model reports its test PSNR and its number of parameters.

Requires scikit-image (marching cubes): pip install scikit-image

Usage (from the repository root):
    python examples/train_gs2mesh_toy.py
    python examples/train_gs2mesh_toy.py --n-gaussians 20000 --grid 48

It saves to --out: test views (ground truth | stage 1 Gaussians | extracted mesh
| trained mesh | hybrid), the hybrid decomposition, and the extracted and trained
meshes (OBJ). Example 7 (notebook) runs the same pipeline step by step.
"""

import argparse
import math
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmesh.scenes.mesh import Mesh
from gmesh.scenes.cameras import OrbitCamera
from gmesh.renderer.rasterizer_mesh import face_adjacency
from gmesh.utils.graphics_utils import compute_vertex_normals
from train_textured_toy import GAUSSIAN_LRS, SH_C0, load_scene, orbit_poses, make_gaussians, to_uint8
from train_hybrid_toy import (REGULARIZER_WEIGHTS, SMOOTHNESS, VERTEX_LR, fit, initial_gaussian_params,
                              large_steps_matrices, make_renderers, mesh_edges, mesh_regularizers,
                              chamfer_distance, save_obj)

COLOR_LR = 2e-2


# ---------------------------------------------------------------------------
# Stage 1: Gaussians from a random initialization
# ---------------------------------------------------------------------------

def random_gaussian_params(n, half_size=1.0, device="cuda", seed=0):
    """n Gaussians uniformly in the cube [-half_size, half_size]^3: gray, opacity 0.1, sized to fill it."""
    g = torch.Generator().manual_seed(seed)
    spacing = 2 * half_size / n ** (1 / 3)
    return {
        "means": ((torch.rand(n, 3, generator=g) * 2 - 1) * half_size).to(device),
        "quats": F.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device),
        "log_scales": torch.full((n, 3), math.log(0.5 * spacing), device=device),
        "opacity_logits": torch.full((n,), math.log(0.1 / 0.9), device=device),  # opacity 0.1
        "sh": torch.zeros(n, 1, 3, device=device),                               # gray
    }


def train_gaussians(renderers, g_params, gt_train, gt_test, train_poses, test_poses, iters, **kw):
    gaussian_rasterizer = renderers[1]

    def render(pose):
        rgb, _, alpha = gaussian_rasterizer.rasterize(make_gaussians(g_params), pose)
        return rgb[0], alpha[0]

    return fit(render, g_params, GAUSSIAN_LRS, gt_train, gt_test, train_poses, test_poses, iters, **kw)


# ---------------------------------------------------------------------------
# Stage 2: mesh extraction (TSDF fusion of the Gaussians' depth + marching cubes)
# ---------------------------------------------------------------------------

def extraction_poses(device, distance=3.0):
    """60 orbit views on 5 rings of elevation"""
    return [torch.tensor([[elev, 2 * math.pi * k / 12 + 0.3 * i, distance]], device=device)
            for i, elev in enumerate([-0.6, -0.3, 0.0, 0.3, 0.6]) for k in range(12)]


@torch.no_grad()
def extract_mesh(gaussian_rasterizer, camera, g_params, poses, grid=48, alpha_threshold=0.5, opacity_threshold=0.5,
                 margin=0.1):
    """
    Fuses the depth maps of the Gaussians (where their alpha > alpha_threshold; free space elsewhere) into a
    truncated signed distance volume over the bounding box of the opaque Gaussians, and extracts its zero
    level set with marching cubes. Returns verts [V, 3], faces [F, 3] (outward facing, largest component).
    """
    from skimage.measure import marching_cubes
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    gaussians = make_gaussians(g_params)
    device = gaussians.means.device
    opaque = gaussians.means[gaussians.opacities > opacity_threshold]
    lo, hi = opaque.amin(dim=0) - margin, opaque.amax(dim=0) + margin
    voxel = ((hi - lo).max() / (grid - 1)).item()
    dims = [int(math.ceil((hi - lo)[k].item() / voxel)) + 1 for k in range(3)]
    axes = [lo[k] + voxel * torch.arange(dims[k], device=device) for k in range(3)]
    points = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    trunc = 3.0 * voxel

    sdf_sum = torch.zeros(points.shape[0], device=device)
    weight = torch.zeros(points.shape[0], device=device)
    H, W = camera.image_height, camera.image_width
    for pose in poses:
        _, depth, alpha = gaussian_rasterizer.rasterize(gaussians, pose)
        V = camera.get_view_matrix(pose)[0]
        K = camera.get_intrinsic_matrix(batch_size=1)[0]
        pc = points @ V[:3, :3].T + V[:3, 3]
        z = pc[:, 2]
        u = K[0, 0] * pc[:, 0] / z.clamp(min=1e-6) + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z.clamp(min=1e-6) + K[1, 2]
        px, py = u.floor().long(), v.floor().long()
        seen = (z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        a = torch.zeros_like(z)
        d = torch.zeros_like(z)
        a[seen] = alpha[0, py[seen], px[seen], 0]
        d[seen] = depth[0, py[seen], px[seen], 0]
        surface = seen & (a > alpha_threshold)
        sdf = torch.where(surface, (d - z).clamp(max=trunc), torch.full_like(z, trunc))  # free space: outside
        use = seen & (~surface | (d - z > -trunc))                                         # not far behind the surface
        sdf_sum += torch.where(use, sdf, torch.zeros_like(sdf))
        weight += use.float()
    sdf = torch.where(weight > 0, sdf_sum / weight.clamp(min=1), torch.full_like(sdf_sum, trunc)).view(*dims)

    verts, faces, _, _ = marching_cubes(sdf.cpu().numpy(), level=0.0, spacing=(voxel,) * 3)
    verts = torch.tensor(verts, dtype=torch.float32, device=device) + lo
    faces = torch.tensor(faces.copy(), dtype=torch.int64, device=device)

    # largest connected component
    f = faces.cpu().numpy()
    n = verts.shape[0]
    graph = coo_matrix((np.ones(3 * len(f)), (np.concatenate([f[:, 0], f[:, 1], f[:, 2]]),
                                                np.concatenate([f[:, 1], f[:, 2], f[:, 0]]))), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    keep = labels == np.bincount(labels).argmax()
    remap = -np.ones(n, dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    f = remap[f]
    f = f[(f >= 0).all(axis=1)]
    verts, faces = verts[torch.from_numpy(keep).to(device)], torch.from_numpy(f).to(device)

    # outward facing
    v0, v1, v2 = (verts[faces[:, k]] for k in range(3))
    if (torch.cross(v1 - v0, v2 - v0, dim=-1) * (v0 + v1 + v2 - 3 * verts.mean(dim=0))).sum() < 0:
        faces = faces[:, [0, 2, 1]]
    return verts, faces


@torch.no_grad()
def colors_from_gaussians(verts, faces, g_params, light, opacity_threshold=0.5):
    """
    Vertex colors from the nearest opaque Gaussian. The Gaussians' colors include the lighting, the mesh colors
    are base colors (lit by the renderer), so the diffuse shading of the light at each vertex is divided out.
    """
    gaussians = make_gaussians(g_params)
    opaque = gaussians.opacities > opacity_threshold
    means = gaussians.means[opaque]
    rgb = (0.5 + SH_C0 * gaussians.colors[opaque, 0]).clamp(0.02, 0.98)
    nearest = torch.cat([torch.cdist(c, means).argmin(dim=1) for c in verts.split(4096)])
    normals = compute_vertex_normals(verts, faces)
    to_light = F.normalize(light.position - verts, dim=-1)
    shading = light.ambient + light.diffuse * (normals * to_light).sum(dim=-1, keepdim=True).clamp(min=0)
    return (rgb[nearest] / shading.clamp(min=0.05)).clamp(0.02, 0.98)


# ---------------------------------------------------------------------------
# Stages 3 and 4: mesh, then mesh + Gaussians
# ---------------------------------------------------------------------------

def initial_mesh(verts, faces, colors, smoothness=SMOOTHNESS):
    """a mesh to train: dict with its faces, large steps matrices, and parameters (vertex_u, color_logits)"""
    M, M_inv = large_steps_matrices(verts.shape[0], faces, smoothness)
    return {"faces": faces, "M": M, "M_inv": M_inv,
            "params": {"vertex_u": M @ verts, "color_logits": torch.logit(colors.clamp(0.02, 0.98))}}


def mesh_verts(mesh, params=None):
    return mesh["M_inv"] @ (params or mesh["params"])["vertex_u"]


def make_mesh(mesh, params=None):
    params = params or mesh["params"]
    return Mesh(verts=mesh_verts(mesh, params), faces=mesh["faces"], colors=torch.sigmoid(params["color_logits"]))


def _regularizer(mesh, params):
    faces = mesh["faces"]
    edges, adjacency = mesh_edges(faces), face_adjacency(faces).long()

    def regularize():
        reg = mesh_regularizers(mesh_verts(mesh, params), faces, edges, adjacency)
        return sum(REGULARIZER_WEIGHTS[k] * v for k, v in reg.items())
    return regularize


def train_mesh(renderers, mesh, gt_train, gt_test, train_poses, test_poses, iters, **kw):
    mesh_renderer = renderers[0]
    params = mesh["params"]

    def render(pose):
        rgb, _, alpha = mesh_renderer.render(make_mesh(mesh, params), pose)
        return rgb[0], alpha[0]

    lrs = {"vertex_u": VERTEX_LR, "color_logits": COLOR_LR}
    return fit(render, params, lrs, gt_train, gt_test, train_poses, test_poses, iters,
               regularize=_regularizer(mesh, params), **kw)


def train_hybrid(renderers, mesh, n_gaussians, gt_train, gt_test, train_poses, test_poses, iters, **kw):
    """Adds n Gaussians on the mesh and trains both. Returns (history, mesh params, Gaussian params)."""
    gmesh_renderer = renderers[2]
    m_params = {k: v.detach().clone() for k, v in mesh["params"].items()}
    g_params = initial_gaussian_params(mesh_verts(mesh, m_params), mesh["faces"], n_gaussians)

    def render(pose):
        rgb, _, alpha = gmesh_renderer.render(mesh=make_mesh(mesh, m_params), gaussians=make_gaussians(g_params),
                                              camera_pose=pose)
        return rgb[0], alpha[0]

    lrs = {"vertex_u": 0.2 * VERTEX_LR, "color_logits": 0.2 * COLOR_LR, **GAUSSIAN_LRS}
    hist = fit(render, {**m_params, **g_params}, lrs, gt_train, gt_test, train_poses, test_poses, iters,
               regularize=_regularizer(mesh, m_params), **kw)
    return hist, m_params, g_params


def count(params):
    return sum(p.numel() for p in params.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-gaussians", type=int, default=20000, help="N: Gaussians of stage 1 (stage 4 adds N / 4)")
    parser.add_argument("--gaussian-iters", type=int, default=3000, help="stage 1 iterations")
    parser.add_argument("--mesh-iters", type=int, default=1500, help="stage 3 iterations")
    parser.add_argument("--hybrid-iters", type=int, default=2000, help="stage 4 iterations")
    parser.add_argument("--grid", type=int, default=48, help="resolution of the signed distance volume")
    parser.add_argument("--size", type=int, default=256, help="image width and height")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=os.path.join(ROOT, "output", "gs2mesh_toy_training"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = args.device
    gt_mesh, light = load_scene(device)
    camera = OrbitCamera(image_width=args.size, image_height=args.size, fov_x=40, fov_y=40,
                         z_near=0.1, z_far=100, device=device)
    renderers = make_renderers(camera, light)
    train_poses, test_poses = orbit_poses(device)
    with torch.no_grad():
        gt_train = [(r[0], a[0]) for r, _, a in (renderers[0].render(gt_mesh, p) for p in train_poses)]
        gt_test = [(r[0], a[0]) for r, _, a in (renderers[0].render(gt_mesh, p) for p in test_poses)]
    n, n_small = args.n_gaussians, args.n_gaussians // 4
    later_iters = args.mesh_iters + args.hybrid_iters

    print(f"\n=== stage 1: {n} Gaussians from a random initialization ===")
    g1 = random_gaussian_params(n, device=device)
    hist1 = train_gaussians(renderers, g1, gt_train, gt_test, train_poses, test_poses, args.gaussian_iters,
                            log_every=args.log_every)

    print("\n=== stage 2: mesh extraction ===")
    verts, faces = extract_mesh(renderers[1], camera, g1, extraction_poses(device), grid=args.grid)
    print(f"extracted mesh: {verts.shape[0]} vertices, {faces.shape[0]} triangles; Chamfer distance to the ground "
          f"truth {chamfer_distance(verts, faces, gt_mesh.verts, gt_mesh.faces):.4f}")
    save_obj(os.path.join(args.out, "extracted_mesh.obj"), verts, faces)
    mesh = initial_mesh(verts, faces, colors_from_gaussians(verts, faces, g1, light))
    with torch.no_grad():
        extracted_views = [renderers[0].render(make_mesh(mesh), p)[0][0] for p in test_poses]

    print("\n=== stage 3: mesh ===")
    hist3 = train_mesh(renderers, mesh, gt_train, gt_test, train_poses, test_poses, args.mesh_iters,
                       log_every=args.log_every)

    print(f"\n=== stage 4: mesh + {n_small} Gaussians ===")
    hist4, m_params, g4 = train_hybrid(renderers, mesh, n_small, gt_train, gt_test, train_poses, test_poses,
                                       args.hybrid_iters, log_every=args.log_every)
    save_obj(os.path.join(args.out, "trained_mesh.obj"), mesh_verts(mesh, m_params).detach(), faces)

    print(f"\n=== baseline: the stage 1 Gaussians, {later_iters} more iterations ===")
    g1_more = {k: v.detach().clone() for k, v in g1.items()}
    hist_more = train_gaussians(renderers, g1_more, gt_train, gt_test, train_poses, test_poses, later_iters,
                                log_every=args.log_every)
    print(f"\n=== baseline: {n_small} Gaussians alone ===")
    g_small = random_gaussian_params(n_small, device=device, seed=1)
    hist_small = train_gaussians(renderers, g_small, gt_train, gt_test, train_poses, test_poses,
                                 args.gaussian_iters + later_iters, log_every=args.log_every)

    print("\n=== summary ===")
    rows = [
        (f"stage 1: {n} Gaussians", hist1, count(g1)),
        ("stage 3: mesh", hist3, count(mesh["params"])),
        (f"stage 4: mesh + {n_small} Gaussians", hist4, count(m_params) + count(g4)),
        (f"{n} Gaussians, {later_iters} more iterations", hist_more, count(g1_more)),
        (f"{n_small} Gaussians alone", hist_small, count(g_small)),
    ]
    for name, h, p in rows:
        print(f"{name:42s} test PSNR {h['psnr'][-1]:6.2f} dB   {p:8d} parameters")
    print(f"mesh Chamfer distance to the ground truth: extracted "
          f"{chamfer_distance(verts, faces, gt_mesh.verts, gt_mesh.faces):.4f}, after stage 4 "
          f"{chamfer_distance(mesh_verts(mesh, m_params).detach(), faces, gt_mesh.verts, gt_mesh.faces):.4f}")

    # test views: ground truth | stage 1 Gaussians | extracted mesh | trained mesh | hybrid
    images = [[gt_test[v][0] for v in range(len(test_poses))], hist1["snapshots"][-1][1], extracted_views,
              hist3["snapshots"][-1][1], hist4["snapshots"][-1][1]]
    grid_rows = [np.concatenate([to_uint8(col[v]) for col in images], axis=1) for v in range(len(test_poses))]
    imageio.imwrite(os.path.join(args.out, "test_views_gt_gaussians_extracted_mesh_hybrid.png"),
                    np.concatenate(grid_rows, axis=0))
    with torch.no_grad():
        grid_rows = []
        for pose in test_poses[:3]:
            ret = renderers[2].render(mesh=make_mesh(mesh, m_params), gaussians=make_gaussians(g4), camera_pose=pose,
                                      return_all=True)
            grid_rows.append(np.concatenate([to_uint8(ret[key][0][0]) for key in ("mesh", "3dgs", "blend")], axis=1))
    imageio.imwrite(os.path.join(args.out, "hybrid_mesh_gaussians_both.png"), np.concatenate(grid_rows, axis=0))
    print(f"\nresults saved to {args.out}")


if __name__ == "__main__":
    main()
