#
# Peizhi Yan
# Copyright (C) 2026
#

"""
Hybrid toy example: a coarse textured mesh plus a few Gaussians reproduce a
detailed object better than either alone.

Ground truth: the detailed textured Spot mesh (5856 triangles, assets/data/spot),
lit by a point light, pre-rendered from 24 orbit views (training) and 6 other
views (testing).

Models (trained with Adam and an L1 loss on the same images):

  mesh:       the coarse Spot control mesh (360 triangles) with a learned
              texture map and learned vertex positions
  gaussians:  K 3D Gaussians alone (plain 3DGS), initialized on the coarse mesh
  hybrid:     the coarse textured mesh and K Gaussians, rendered together with
              Gmesh: the mesh carries the large textured surfaces, the Gaussians
              the detail it misses (round shapes, ears, horns, ...), in front of
              or behind its surface, and both are trained jointly

For several Gaussian budgets K, it compares the test PSNR of the three models
(the mesh model has no Gaussians).

Usage (from the repository root):
    python examples/train_hybrid_toy.py
    python examples/train_hybrid_toy.py --budgets 1000 4000 --iters 2000 --size 256

It prints the test PSNR and the number of parameters of each run, and saves to
--out the PSNR vs. Gaussian budget plot, test views (ground truth | mesh |
Gaussians | hybrid), and the decomposition of the hybrid model (its mesh alone,
its Gaussians alone). Example 6 (notebook) runs the same experiment step by step.
"""

import argparse
import math
import os
import sys

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
from gmesh.renderer.rasterizer_mesh import MeshRenderer
from gmesh.renderer.rasterizer_gaussians import GaussianRasterizer
from gmesh.renderer.gmesh_renderer import GmeshRenderer
from gmesh.utils.mesh_io import load_obj
from train_textured_toy import (SPOT_DIR, GAUSSIAN_LRS, TEXTURE_LR, load_scene, orbit_poses, make_gaussians,
                                train_loop, to_uint8)

VERTEX_LR = 2e-3


# ---------------------------------------------------------------------------
# Scene and models
# ---------------------------------------------------------------------------

def load_coarse_mesh(device):
    """The Spot control mesh (quads split into 360 triangles, with texture coordinates), rotated as the ground truth."""
    verts, faces, uvs, faces_uv = load_obj(os.path.join(SPOT_DIR, "spot_control_mesh.obj"), device=device)
    theta = math.pi * 0.8
    rotation = torch.tensor([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)],
    ], dtype=torch.float32, device=device)
    return verts @ rotation.T, faces, uvs, faces_uv


def sample_surface(verts, faces, n, seed=0):
    """n points uniformly distributed (by area) on a triangle mesh. Returns [n, 3] and the total area."""
    g = torch.Generator().manual_seed(seed)
    v0, v1, v2 = (verts[faces[:, k]] for k in range(3))
    areas = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)
    face = torch.multinomial(areas.cpu(), n, replacement=True, generator=g).to(verts.device)
    r1, r2 = torch.rand(2, n, generator=g).to(verts.device)
    s = r1.sqrt()
    b0, b1, b2 = 1 - s, s * (1 - r2), s * r2
    points = b0[:, None] * v0[face] + b1[:, None] * v1[face] + b2[:, None] * v2[face]
    return points, areas.sum().item()


def initial_mesh_params(coarse, tex_size):
    verts, faces, uvs, faces_uv = coarse
    return {
        "vertex_offsets": torch.zeros_like(verts),
        "texture_logits": torch.zeros(tex_size, tex_size, 3, device=verts.device),  # gray
    }


def make_mesh(coarse, p):
    verts, faces, uvs, faces_uv = coarse
    return Mesh(verts=verts + p["vertex_offsets"], faces=faces, uvs=uvs, faces_uv=faces_uv,
                texture=torch.sigmoid(p["texture_logits"]))


def initial_gaussian_params(coarse, n, seed=0):
    """n Gaussians on the coarse mesh surface: random rotations, gray, half transparent, sized to cover it."""
    verts, faces, _, _ = coarse
    means, area = sample_surface(verts, faces, n, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    scale = 0.5 * math.sqrt(area / n)
    device = verts.device
    return {
        "means": means,
        "quats": F.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device),
        "log_scales": torch.full((n, 3), math.log(scale), device=device),
        "opacity_logits": torch.zeros(n, device=device),  # opacity 0.5
        "sh": torch.zeros(n, 1, 3, device=device),        # gray (rgb 0.5)
    }


def count_parameters(params):
    return sum(p.numel() for p in params.values())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, renderers, coarse, n_gaussians, gt_train, gt_test, train_poses, test_poses, iters,
                tex_size=256, snapshot_every=None, log_every=200, seed=0):
    """
    Trains one model ("mesh", "gaussians" or "hybrid"). Returns (history, mesh params, Gaussian params);
    the history also has the number of trained parameters ('n_params').
    """
    torch.manual_seed(seed)
    mesh_renderer, gaussian_rasterizer, gmesh_renderer = renderers
    m_params = initial_mesh_params(coarse, tex_size) if model in ("mesh", "hybrid") else {}
    g_params = initial_gaussian_params(coarse, n_gaussians, seed=seed) if model in ("gaussians", "hybrid") else {}
    for p in list(m_params.values()) + list(g_params.values()):
        p.requires_grad_(True)

    lrs = {"vertex_offsets": VERTEX_LR, "texture_logits": TEXTURE_LR, **GAUSSIAN_LRS}
    params = {**m_params, **g_params}
    optimizer = torch.optim.Adam([{"params": [p], "lr": lrs[k], "name": k} for k, p in params.items()], eps=1e-15)

    def render(pose):
        if model == "mesh":
            rgb, _, _ = mesh_renderer.render(make_mesh(coarse, m_params), pose)
        elif model == "gaussians":
            rgb, _, _ = gaussian_rasterizer.rasterize(make_gaussians(g_params), pose)
        else:
            rgb, _, _ = gmesh_renderer.render(mesh=make_mesh(coarse, m_params), gaussians=make_gaussians(g_params),
                                              camera_pose=pose)
        return rgb[0]

    hist = train_loop(render, list(params.values()), optimizer, gt_train, gt_test, train_poses, test_poses, iters,
                      snapshot_every, log_every)
    hist["n_params"] = count_parameters(params)
    return hist, m_params, g_params


def make_renderers(camera, light):
    bg = (1.0, 1.0, 1.0)
    return (MeshRenderer(camera=camera, bg_color=bg, light=light),
            GaussianRasterizer(camera=camera, bg_color=bg),
            GmeshRenderer(camera=camera, bg_color=bg, light=light))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budgets", type=int, nargs="+", default=[500, 2000, 8000], help="numbers of Gaussians")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--size", type=int, default=256, help="image width and height")
    parser.add_argument("--tex-size", type=int, default=256, help="resolution of the learned texture map")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=os.path.join(ROOT, "output", "hybrid_toy_training"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = args.device
    gt_mesh, light = load_scene(device)
    coarse = load_coarse_mesh(device)
    print(f"ground truth: {gt_mesh.num_faces} triangles; coarse mesh: {coarse[1].shape[0]} triangles")
    camera = OrbitCamera(image_width=args.size, image_height=args.size, fov_x=40, fov_y=40,
                         z_near=0.1, z_far=100, device=device)
    renderers = make_renderers(camera, light)
    train_poses, test_poses = orbit_poses(device)
    with torch.no_grad():
        gt_train = [renderers[0].render(gt_mesh, p)[0][0] for p in train_poses]
        gt_test = [renderers[0].render(gt_mesh, p)[0][0] for p in test_poses]

    runs = [("mesh", 0)] + [(m, k) for k in args.budgets for m in ("gaussians", "hybrid")]
    results = {}
    for model, k in runs:
        print(f"\n=== {model}" + (f", {k} Gaussians" if k else "") + " ===")
        hist, m_params, g_params = train_model(model, renderers, coarse, k, gt_train, gt_test, train_poses,
                                               test_poses, args.iters, tex_size=args.tex_size,
                                               log_every=args.log_every)
        results[(model, k)] = (hist, m_params, g_params)
        print(f"test PSNR {hist['psnr'][0]:.2f} -> {hist['psnr'][-1]:.2f} dB, {hist['n_params']} parameters")

    # summary and plot
    print("\n=== summary (test PSNR) ===")
    print(f"{'Gaussians':>10s} {'mesh':>8s} {'gaussians':>10s} {'hybrid':>8s}")
    mesh_psnr = results[("mesh", 0)][0]["psnr"][-1]
    for k in args.budgets:
        print(f"{k:10d} {mesh_psnr:8.2f} {results[('gaussians', k)][0]['psnr'][-1]:10.2f} "
              f"{results[('hybrid', k)][0]['psnr'][-1]:8.2f}")
    fig = Figure(figsize=(5.5, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(mesh_psnr, color="gray", linestyle="--", label="coarse mesh only")
    for model in ("gaussians", "hybrid"):
        ax.plot(args.budgets, [results[(model, k)][0]["psnr"][-1] for k in args.budgets], "o-",
                label="Gaussians only" if model == "gaussians" else "hybrid (coarse mesh + Gaussians)")
    ax.set_xscale("log")
    ax.set_xlabel("number of Gaussians")
    ax.set_ylabel("test PSNR (dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "psnr_vs_gaussians.png"), dpi=120)

    # test views: ground truth | mesh | Gaussians | hybrid (middle budget)
    k = args.budgets[len(args.budgets) // 2]
    finals = [results[("mesh", 0)][0], results[("gaussians", k)][0], results[("hybrid", k)][0]]
    rows = [np.concatenate([to_uint8(gt_test[v])] + [to_uint8(h["snapshots"][-1][1][v]) for h in finals], axis=1)
            for v in range(len(test_poses))]
    imageio.imwrite(os.path.join(args.out, f"test_views_gt_mesh_gaussians_hybrid_{k}.png"),
                    np.concatenate(rows, axis=0))

    # decomposition of the hybrid model: its mesh alone, its Gaussians alone, both
    _, m_params, g_params = results[("hybrid", k)]
    with torch.no_grad():
        rows = []
        for pose in test_poses[:3]:
            ret = renderers[2].render(mesh=make_mesh(coarse, m_params), gaussians=make_gaussians(g_params),
                                      camera_pose=pose, return_all=True)
            rows.append(np.concatenate([to_uint8(ret[key][0][0]) for key in ("mesh", "3dgs", "blend")], axis=1))
    imageio.imwrite(os.path.join(args.out, f"hybrid_{k}_mesh_gaussians_both.png"), np.concatenate(rows, axis=0))
    print(f"\nresults saved to {args.out}")


if __name__ == "__main__":
    main()
