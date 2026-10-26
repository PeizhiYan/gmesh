#
# Peizhi Yan
# Copyright (C) 2026
#

"""
Hybrid toy example: reconstruct an object as a textured mesh, starting from a
sphere, then add Gaussians for what the mesh misses, and fine-tune both together.

Ground truth: the detailed textured Spot mesh (5856 triangles, assets/data/spot),
lit by a point light, pre-rendered (color and silhouette) from 24 orbit views
(training) and 6 other views (testing).

  stage 1 (mesh):    an icosphere (5120 triangles, sphere-mapped texture
                     coordinates) is deformed into the object, and its texture
                     map learned, from the color and silhouette images (L1).
                     The vertices are optimized in the "large steps"
                     parameterization u = (I + lambda L) v (Nicolet et al.,
                     "Large Steps in Inverse Rendering of Geometry", 2021),
                     which smooths the updates over the surface, with light
                     smoothness regularizers (Laplacian, edge length, normal
                     consistency)
  stage 2 (hybrid):  K Gaussians are added on the fitted mesh, and the mesh
                     (vertices, texture) and the Gaussians are fine-tuned
                     together through Gmesh: the mesh carries the large
                     textured surfaces, the Gaussians the parts a deformed
                     sphere hardly reaches (ears, horns, thin legs), in front
                     of or behind its surface
  baselines:         K Gaussians alone (plain 3DGS), (a) initialized on the
                     sphere and trained for as many iterations as both stages,
                     (b) initialized on the stage 1 mesh (same start as the
                     hybrid model) and trained for as many iterations as stage 2

It reports the test PSNR of the three models and the Chamfer distance of the
meshes to the ground-truth mesh.

Usage (from the repository root):
    python examples/train_hybrid_toy.py
    python examples/train_hybrid_toy.py --mesh-iters 2000 --hybrid-iters 2000 --n-gaussians 4000

It saves to --out: test views (ground truth | mesh | Gaussians (a) | Gaussians (b) | hybrid), the
decomposition of the hybrid model (its mesh, its Gaussians, both), the mesh
during stage 1, loss curves, and the fitted mesh (OBJ). Example 6 (notebook)
runs the same experiment step by step.
"""

import argparse
import math
import os
import sys
import time

import imageio.v2 as imageio
import numpy as np
from matplotlib.figure import Figure  # no pyplot: importing this file must not change the plotting backend
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmesh.scenes.mesh import Mesh
from gmesh.scenes.cameras import OrbitCamera
from gmesh.renderer.rasterizer_mesh import MeshRenderer, face_adjacency
from gmesh.renderer.rasterizer_gaussians import GaussianRasterizer
from gmesh.renderer.gmesh_renderer import GmeshRenderer
from train_textured_toy import GAUSSIAN_LRS, TEXTURE_LR, load_scene, orbit_poses, make_gaussians, psnr, to_uint8

VERTEX_LR = 2e-2            # learning rate of the vertex parameters u (large steps parameterization)
SMOOTHNESS = 10.0           # lambda of u = (I + lambda L) v
SILHOUETTE_WEIGHT = 1.0
REGULARIZER_WEIGHTS = {"laplacian": 0.05, "edge": 0.02, "normal": 0.01}


# ---------------------------------------------------------------------------
# Sphere mesh
# ---------------------------------------------------------------------------

def icosphere(level=4, device="cpu"):
    """Unit icosphere: 10 * 4^level + 2 vertices, 20 * 4^level triangles (outward facing)."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
             [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
             [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
             [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    verts = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in verts]
    for _ in range(level):
        cache, new_faces = {}, []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    verts = torch.tensor(np.array(verts), dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    # outward facing (normals from the face winding, as the shading uses)
    v0, v1, v2 = (verts[faces[:, k]] for k in range(3))
    if (torch.cross(v1 - v0, v2 - v0, dim=-1) * (v0 + v1 + v2)).sum() < 0:
        faces = faces[:, [0, 2, 1]]
    return verts, faces


def sphere_uvs(verts, faces):
    """
    Texture coordinates per face corner from the longitude / latitude of the unit sphere vertices: [3F, 2] and
    faces_uv [F, 3]. Faces crossing the longitude seam get continuous u (rescaled into [0, 1]); the poles take
    the longitude of the face.
    """
    x, y, z = verts.unbind(-1)
    u = 0.5 + torch.atan2(z, x) / (2 * math.pi)
    v = 0.5 + torch.asin(y.clamp(-1, 1)) / math.pi
    fu, fv = u[faces], v[faces]                                     # [F, 3]
    wrap = (fu.max(dim=1).values - fu.min(dim=1).values) > 0.5
    fu = torch.where(wrap[:, None] & (fu < 0.5), fu + 1.0, fu)     # continuous across the seam
    pole = (y[faces].abs() > 1 - 1e-6)
    others = torch.where(pole, torch.zeros_like(fu), fu).sum(dim=1, keepdim=True) / (~pole).sum(dim=1, keepdim=True)
    fu = torch.where(pole, others.expand_as(fu), fu)
    fu = fu / fu.max()
    uvs = torch.stack([fu, fv], dim=-1).reshape(-1, 2)
    faces_uv = torch.arange(uvs.shape[0], device=verts.device).view(-1, 3)
    return uvs, faces_uv


def initial_sphere(gt_verts, level=4, smoothness=SMOOTHNESS):
    """
    An icosphere around the object (center and radius from its bounding box) with sphere-mapped uvs.
    Returns a dict: verts, faces, uvs, faces_uv, and the matrices M = I + smoothness L (L: combinatorial
    Laplacian) and its inverse, for the large steps parameterization of the vertices (u = M v).
    """
    unit, faces = icosphere(level, device=gt_verts.device)
    lo, hi = gt_verts.amin(dim=0), gt_verts.amax(dim=0)
    center, radius = (lo + hi) / 2, 0.5 * (hi - lo).max().item()
    uvs, faces_uv = sphere_uvs(unit, faces)
    M, M_inv = large_steps_matrices(unit.shape[0], faces, smoothness)
    return {"verts": center + radius * unit, "faces": faces, "uvs": uvs, "faces_uv": faces_uv,
            "M": M, "M_inv": M_inv}


def large_steps_matrices(n_verts, faces, smoothness=SMOOTHNESS):
    """M = I + smoothness L (L: combinatorial Laplacian of the mesh) and its inverse (dense)."""
    edges = mesh_edges(faces)
    adjacency = torch.zeros(n_verts, n_verts, device=faces.device)
    adjacency[edges[:, 0], edges[:, 1]] = 1.0
    adjacency[edges[:, 1], edges[:, 0]] = 1.0
    laplacian = torch.diag(adjacency.sum(dim=1)) - adjacency
    M = torch.eye(n_verts, device=faces.device) + smoothness * laplacian
    return M, torch.linalg.inv(M)


# ---------------------------------------------------------------------------
# Regularizers and geometry error
# ---------------------------------------------------------------------------

def mesh_edges(faces):
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).sort(dim=1).values
    return torch.unique(e, dim=0)


def mesh_regularizers(verts, faces, edges, adjacency):
    """Scale-free smoothness terms: uniform Laplacian, edge length variance, normal consistency."""
    e0, e1 = verts[edges[:, 0]], verts[edges[:, 1]]
    length = (e0 - e1).norm(dim=-1)
    mean_len = length.mean().detach()

    neighbor_sum = torch.zeros_like(verts).index_add(0, edges[:, 0], e1).index_add(0, edges[:, 1], e0)
    degree = torch.zeros(verts.shape[0], device=verts.device).index_add(
        0, edges.flatten(), torch.ones(edges.numel(), device=verts.device))
    laplacian = verts - neighbor_sum / degree[:, None]

    v0, v1, v2 = (verts[faces[:, k]] for k in range(3))
    n = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    valid = adjacency >= 0
    cos = (n[:, None, :] * n[adjacency.clamp(min=0)]).sum(dim=-1)

    return {
        "laplacian": (laplacian ** 2).sum(dim=-1).mean() / mean_len ** 2,
        "edge": ((length - mean_len) ** 2).mean() / mean_len ** 2,
        "normal": (1.0 - cos)[valid].mean(),
    }


def sample_surface(verts, faces, n, seed=0):
    """n points uniformly distributed (by area) on a triangle mesh. Returns [n, 3] and the total area."""
    g = torch.Generator().manual_seed(seed)
    v0, v1, v2 = (verts[faces[:, k]] for k in range(3))
    areas = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)
    face = torch.multinomial(areas.detach().cpu(), n, replacement=True, generator=g).to(verts.device)
    r1, r2 = torch.rand(2, n, generator=g).to(verts.device)
    s = r1.sqrt()
    b0, b1, b2 = 1 - s, s * (1 - r2), s * r2
    points = b0[:, None] * v0[face] + b1[:, None] * v1[face] + b2[:, None] * v2[face]
    return points, areas.sum().item()


@torch.no_grad()
def chamfer_distance(verts_a, faces_a, verts_b, faces_b, n=20000):
    """Symmetric Chamfer distance (mean nearest-point distance, both ways) between two meshes' surfaces."""
    a, _ = sample_surface(verts_a, faces_a, n, seed=0)
    b, _ = sample_surface(verts_b, faces_b, n, seed=1)

    def nearest(p, q):
        return torch.cat([torch.cdist(c, q).min(dim=1).values for c in p.split(4096)])
    return 0.5 * (nearest(a, b).mean() + nearest(b, a).mean()).item()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def initial_mesh_params(sphere, tex_size):
    """Vertex parameters u = M v (large steps) of the sphere, and a gray texture."""
    return {"vertex_u": sphere["M"] @ sphere["verts"],
            "texture_logits": torch.zeros(tex_size, tex_size, 3, device=sphere["verts"].device)}


def mesh_verts(sphere, p):
    """vertex positions v = M^-1 u"""
    return sphere["M_inv"] @ p["vertex_u"]


def make_mesh(sphere, p):
    return Mesh(verts=mesh_verts(sphere, p), faces=sphere["faces"], uvs=sphere["uvs"], faces_uv=sphere["faces_uv"],
                texture=torch.sigmoid(p["texture_logits"]))


def initial_gaussian_params(verts, faces, n, seed=0):
    """n Gaussians on a mesh surface: random rotations, gray, half transparent, sized to cover it."""
    means, area = sample_surface(verts.detach(), faces, n, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    device = verts.device
    return {
        "means": means,
        "quats": F.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device),
        "log_scales": torch.full((n, 3), math.log(0.5 * math.sqrt(area / n)), device=device),
        "opacity_logits": torch.zeros(n, device=device),  # opacity 0.5
        "sh": torch.zeros(n, 1, 3, device=device),        # gray (rgb 0.5)
    }


def make_renderers(camera, light):
    bg = (1.0, 1.0, 1.0)
    return (MeshRenderer(camera=camera, bg_color=bg, light=light),
            GaussianRasterizer(camera=camera, bg_color=bg),
            GmeshRenderer(camera=camera, bg_color=bg, light=light))


def fit(render, params, lrs, gt_train, gt_test, train_poses, test_poses, iters, regularize=None,
        snapshot_every=None, log_every=200, seed=0):
    """
    Adam on `params` (dict of tensors, learning rates `lrs` decaying 10x over the run) with the loss
    L1(color) + SILHOUETTE_WEIGHT * L1(alpha) [+ regularize()]. `render(pose)` returns (rgb, alpha) [H, W, *].
    gt_train / gt_test are lists of (rgb, alpha). Returns the history: losses, test PSNR, test-view snapshots.
    """
    torch.manual_seed(seed)
    for p in params.values():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam([{"params": [p], "lr": lrs[k]} for k, p in params.items()], eps=1e-15)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1.0 / iters))

    def test_views():
        with torch.no_grad():
            return [render(pose)[0] for pose in test_poses]

    def test_psnr(views):
        return float(np.mean([psnr(v, g[0]) for v, g in zip(views, gt_test)]))

    views = test_views()
    hist = {"loss": [], "psnr_iters": [0], "psnr": [test_psnr(views)], "snapshots": [(0, views)]}
    t0 = time.time()
    for it in range(1, iters + 1):
        k = torch.randint(len(train_poses), (1,)).item()
        rgb, alpha = render(train_poses[k])
        loss = (rgb - gt_train[k][0]).abs().mean() + SILHOUETTE_WEIGHT * (alpha - gt_train[k][1]).abs().mean()
        if regularize is not None:
            loss = loss + regularize()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        hist["loss"].append(loss.item())
        if (snapshot_every and it % snapshot_every == 0) or it == iters:
            views = test_views()
            hist["psnr_iters"].append(it)
            hist["psnr"].append(test_psnr(views))
            hist["snapshots"].append((it, views))
        if log_every and (it % log_every == 0 or it == iters):
            evaluated = f"  test PSNR {hist['psnr'][-1]:.2f}" if hist["psnr_iters"][-1] == it else ""
            print(f"iter {it:5d}  loss {np.mean(hist['loss'][-log_every:]):.4f}{evaluated}  ({time.time() - t0:.1f}s)")
    return hist


def train_mesh_stage(renderers, sphere, gt_train, gt_test, train_poses, test_poses, iters, tex_size=512, **kw):
    """Stage 1: deform the sphere and learn its texture. Returns (history, mesh params)."""
    mesh_renderer = renderers[0]
    faces = sphere["faces"]
    m_params = initial_mesh_params(sphere, tex_size)
    edges, adjacency = mesh_edges(faces), face_adjacency(faces).long()

    def render(pose):
        rgb, _, alpha = mesh_renderer.render(make_mesh(sphere, m_params), pose)
        return rgb[0], alpha[0]

    def regularize():
        reg = mesh_regularizers(mesh_verts(sphere, m_params), faces, edges, adjacency)
        return sum(REGULARIZER_WEIGHTS[k] * v for k, v in reg.items())

    lrs = {"vertex_u": VERTEX_LR, "texture_logits": TEXTURE_LR}
    hist = fit(render, m_params, lrs, gt_train, gt_test, train_poses, test_poses, iters, regularize=regularize, **kw)
    return hist, m_params


def train_hybrid_stage(renderers, sphere, m_params, n_gaussians, gt_train, gt_test, train_poses, test_poses, iters,
                       **kw):
    """Stage 2: add Gaussians on the fitted mesh and fine-tune everything. Returns (history, mesh params, Gaussian params)."""
    gmesh_renderer = renderers[2]
    faces = sphere["faces"]
    m_params = {k: v.detach().clone() for k, v in m_params.items()}
    g_params = initial_gaussian_params(mesh_verts(sphere, m_params), faces, n_gaussians)
    edges, adjacency = mesh_edges(faces), face_adjacency(faces).long()

    def render(pose):
        rgb, _, alpha = gmesh_renderer.render(mesh=make_mesh(sphere, m_params), gaussians=make_gaussians(g_params),
                                              camera_pose=pose)
        return rgb[0], alpha[0]

    def regularize():
        reg = mesh_regularizers(mesh_verts(sphere, m_params), faces, edges, adjacency)
        return sum(REGULARIZER_WEIGHTS[k] * v for k, v in reg.items())

    # the mesh is fine-tuned with smaller learning rates
    lrs = {"vertex_u": 0.2 * VERTEX_LR, "texture_logits": 0.2 * TEXTURE_LR, **GAUSSIAN_LRS}
    hist = fit(render, {**m_params, **g_params}, lrs, gt_train, gt_test, train_poses, test_poses, iters,
               regularize=regularize, **kw)
    return hist, m_params, g_params


def train_gaussians_only(renderers, verts, faces, n_gaussians, gt_train, gt_test, train_poses, test_poses, iters,
                         **kw):
    """Baseline: Gaussians alone, initialized on the mesh (verts, faces). Returns (history, Gaussian params)."""
    gaussian_rasterizer = renderers[1]
    g_params = initial_gaussian_params(verts.detach(), faces, n_gaussians)

    def render(pose):
        rgb, _, alpha = gaussian_rasterizer.rasterize(make_gaussians(g_params), pose)
        return rgb[0], alpha[0]

    hist = fit(render, g_params, GAUSSIAN_LRS, gt_train, gt_test, train_poses, test_poses, iters, **kw)
    return hist, g_params


def save_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts.tolist():
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in (faces + 1).tolist():
            f.write(f"f {t[0]} {t[1]} {t[2]}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh-iters", type=int, default=2000, help="stage 1 iterations")
    parser.add_argument("--hybrid-iters", type=int, default=2000, help="stage 2 iterations")
    parser.add_argument("--n-gaussians", type=int, default=2000)
    parser.add_argument("--size", type=int, default=256, help="image width and height")
    parser.add_argument("--tex-size", type=int, default=512, help="resolution of the learned texture map")
    parser.add_argument("--sphere-level", type=int, default=4, help="icosphere subdivisions (4: 5120 triangles)")
    parser.add_argument("--smoothness", type=float, default=SMOOTHNESS, help="lambda of the large steps parameterization")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=os.path.join(ROOT, "output", "hybrid_toy_training"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = args.device
    gt_mesh, light = load_scene(device)
    sphere = initial_sphere(gt_mesh.verts, level=args.sphere_level, smoothness=args.smoothness)
    print(f"ground truth: {gt_mesh.num_faces} triangles; sphere: {sphere['faces'].shape[0]} triangles")
    camera = OrbitCamera(image_width=args.size, image_height=args.size, fov_x=40, fov_y=40,
                         z_near=0.1, z_far=100, device=device)
    renderers = make_renderers(camera, light)
    train_poses, test_poses = orbit_poses(device)
    with torch.no_grad():
        gt_train = [tuple(x[0] for x in renderers[0].render(gt_mesh, p)[::2]) for p in train_poses]  # (rgb, alpha)
        gt_test = [tuple(x[0] for x in renderers[0].render(gt_mesh, p)[::2]) for p in test_poses]

    def chamfer(verts):
        return chamfer_distance(verts.detach(), sphere["faces"], gt_mesh.verts, gt_mesh.faces)

    print("\n=== stage 1: mesh (from a sphere) ===")
    snap = max(1, args.mesh_iters // 5)
    hist_mesh, m_params = train_mesh_stage(renderers, sphere, gt_train, gt_test, train_poses, test_poses,
                                           args.mesh_iters, tex_size=args.tex_size, snapshot_every=snap,
                                           log_every=args.log_every)
    print("\n=== stage 2: hybrid (mesh + Gaussians) ===")
    hist_hybrid, m_params2, g_params = train_hybrid_stage(renderers, sphere, m_params, args.n_gaussians, gt_train,
                                                          gt_test, train_poses, test_poses, args.hybrid_iters,
                                                          log_every=args.log_every)
    print("\n=== baseline (a): Gaussians only, from the sphere ===")
    hist_gauss, _ = train_gaussians_only(renderers, sphere["verts"], sphere["faces"], args.n_gaussians, gt_train,
                                         gt_test, train_poses, test_poses, args.mesh_iters + args.hybrid_iters,
                                         log_every=args.log_every)
    print("\n=== baseline (b): Gaussians only, from the stage 1 mesh ===")
    hist_gauss_mesh, _ = train_gaussians_only(renderers, mesh_verts(sphere, m_params), sphere["faces"],
                                              args.n_gaussians, gt_train, gt_test, train_poses, test_poses,
                                              args.hybrid_iters, log_every=args.log_every)

    print("\n=== summary ===")
    print(f"mesh (stage 1)       test PSNR {hist_mesh['psnr'][-1]:6.2f} dB   Chamfer distance "
          f"{chamfer(sphere['verts']):.4f} (sphere) -> {chamfer(mesh_verts(sphere, m_params)):.4f}")
    print(f"hybrid (stage 2)     test PSNR {hist_hybrid['psnr'][-1]:6.2f} dB   "
          f"Chamfer distance {chamfer(mesh_verts(sphere, m_params2)):.4f}")
    print(f"Gaussians only (a)   test PSNR {hist_gauss['psnr'][-1]:6.2f} dB   ({args.n_gaussians} Gaussians from the "
          f"sphere, {args.mesh_iters + args.hybrid_iters} iterations)")
    print(f"Gaussians only (b)   test PSNR {hist_gauss_mesh['psnr'][-1]:6.2f} dB   ({args.n_gaussians} Gaussians from the "
          f"stage 1 mesh, {args.hybrid_iters} iterations)")

    # outputs
    finals = [hist_mesh, hist_gauss, hist_gauss_mesh, hist_hybrid]
    rows = [np.concatenate([to_uint8(gt_test[v][0])] + [to_uint8(h["snapshots"][-1][1][v]) for h in finals], axis=1)
            for v in range(len(test_poses))]
    imageio.imwrite(os.path.join(args.out, "test_views_gt_mesh_gaussiansA_gaussiansB_hybrid.png"),
                    np.concatenate(rows, axis=0))
    imageio.imwrite(os.path.join(args.out, "stage1_mesh_progress.png"),
                    np.concatenate([to_uint8(views[0]) for _, views in hist_mesh["snapshots"]], axis=1))
    with torch.no_grad():
        rows = []
        for pose in test_poses[:3]:
            ret = renderers[2].render(mesh=make_mesh(sphere, m_params2), gaussians=make_gaussians(g_params),
                                      camera_pose=pose, return_all=True)
            rows.append(np.concatenate([to_uint8(ret[key][0][0]) for key in ("mesh", "3dgs", "blend")], axis=1))
    imageio.imwrite(os.path.join(args.out, "hybrid_mesh_gaussians_both.png"), np.concatenate(rows, axis=0))
    save_obj(os.path.join(args.out, "fitted_mesh.obj"), mesh_verts(sphere, m_params2).detach(), sphere["faces"])
    fig = Figure(figsize=(6, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(hist_mesh["loss"] + hist_hybrid["loss"], linewidth=0.6, label="mesh, then hybrid")
    ax.plot(hist_gauss["loss"], linewidth=0.6, label="Gaussians only")
    ax.axvline(args.mesh_iters, color="gray", linestyle="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "loss.png"), dpi=120)
    print(f"\nresults saved to {args.out}")


if __name__ == "__main__":
    main()
