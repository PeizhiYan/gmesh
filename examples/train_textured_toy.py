#
# Peizhi Yan
# Copyright (C) 2026
#

"""
Toy training example with a textured mesh: images of the textured Spot mesh
(assets/data/spot), lit by a point light, are pre-rendered from 24 orbit views
(training) and 6 other views (testing). From these images, Adam (L1 loss) fits

  texture:    the texture map of the mesh, starting from a uniform gray texture
              (the geometry and the light are known)
  gaussians:  3D Gaussians that mimic the mesh, without any mesh (plain 3DGS):
              their means, rotations, scales, opacities and colors, starting
              from Gaussians on the mesh surface with noisy positions, gray
  all:        both experiments, one after the other (default)

Usage (from the repository root):
    python examples/train_textured_toy.py
    python examples/train_textured_toy.py --experiment gaussians --iters 2000 --size 256
    python examples/train_textured_toy.py --experiment texture --texture-filter bilinear

It prints the training loss and the test PSNR, and saves to --out, for each
experiment, test views (ground truth | initialization | trained) and the loss
curve, and for the texture experiment the learned texture next to the ground
truth. Example 5 (notebook) runs the same experiments step by step.
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

from gmesh.scenes.mesh import Mesh
from gmesh.scenes.gaussians import Gaussians
from gmesh.scenes.cameras import OrbitCamera
from gmesh.scenes.light import PointLight
from gmesh.renderer.rasterizer_mesh import MeshRenderer
from gmesh.renderer.rasterizer_gaussians import GaussianRasterizer
from gmesh.utils.mesh_io import load_textured_mesh
from gmesh.utils.mesh2gaussians_utils import mesh_to_gaussians

SPOT_DIR = os.path.join(ROOT, "assets", "data", "spot")
SH_C0 = 0.28209479177387814  # rgb = 0.5 + SH_C0 * sh (degree 0)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def load_scene(device):
    """The textured Spot mesh (rotated to face the first camera) and the light."""
    mesh = load_textured_mesh(os.path.join(SPOT_DIR, "spot_triangulated.obj"),
                              os.path.join(SPOT_DIR, "spot_texture.png"), device=device)
    theta = math.pi * 0.8
    rotation = torch.tensor([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)],
    ], dtype=torch.float32, device=device)
    mesh.verts = mesh.verts @ rotation.T
    # diffuse light only: the images do not depend on the view direction (as Gaussians with SH degree 0)
    light = PointLight(position=(1.5, 2.5, 3.0), ambient=(0.45, 0.45, 0.45), diffuse=(0.65, 0.65, 0.65),
                       specular=(0.0, 0.0, 0.0), device=device)
    return mesh, light


def orbit_poses(device):
    train = [torch.tensor([[(0.45 if k % 2 else -0.3), 2 * math.pi * k / 24, 3.0]], device=device) for k in range(24)]
    test = [torch.tensor([[0.15, 2 * math.pi * (k + 0.5) / 6, 3.0]], device=device) for k in range(6)]
    return train, test


def textured_mesh(mesh, texture):
    """the mesh with another texture map"""
    return Mesh(verts=mesh.verts, faces=mesh.faces, uvs=mesh.uvs, faces_uv=mesh.faces_uv, texture=texture)


def initial_gaussians(mesh, device, noise=0.02, seed=0):
    """Raw parameters of Gaussians on the mesh surface (3 per face), with noisy positions, gray, half transparent."""
    g = torch.Generator().manual_seed(seed)
    init = mesh_to_gaussians(Mesh(verts=mesh.verts, faces=mesh.faces), method="facet3")
    n = init.num_gaussians
    return {
        "means": init.means + (noise * torch.randn(n, 3, generator=g)).to(device),
        "quats": F.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device),
        "log_scales": torch.full((n, 3), math.log(0.012), device=device),
        "opacity_logits": torch.zeros(n, device=device),  # opacity 0.5
        "sh": torch.zeros(n, 1, 3, device=device),        # gray (rgb 0.5)
    }


def make_gaussians(p):
    """Raw parameters -> Gaussians (positive scales, opacities in (0, 1))."""
    return Gaussians(
        means=p["means"],
        quats=p["quats"],
        scales=torch.exp(p["log_scales"]),
        opacities=torch.sigmoid(p["opacity_logits"]),
        colors=p["sh"],
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def psnr(a, b):
    mse = torch.mean((a - b) ** 2).clamp(min=1e-10)
    return (-10.0 * torch.log10(mse)).item()


def to_uint8(img):
    return (img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


TEXTURE_LR = 0.05
GAUSSIAN_LRS = {"means": 1e-3, "quats": 5e-3, "log_scales": 5e-3, "opacity_logits": 5e-2, "sh": 1e-2}


def train_texture(mesh_renderer, mesh, light, gt_train, gt_test, train_poses, test_poses, iters, tex_size=512,
                  snapshot_every=None, log_every=100, seed=0):
    """
    Fits the texture map (through a sigmoid, so it stays in [0, 1]) from a uniform gray one.
    Returns a dict with the losses, the test PSNR (and its iterations), test-view snapshots and the texture.
    """
    torch.manual_seed(seed)
    device = mesh.verts.device
    logits = torch.zeros(tex_size, tex_size, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=TEXTURE_LR)

    def render(pose):
        rgb, _, _ = mesh_renderer.render(textured_mesh(mesh, torch.sigmoid(logits)), pose, light=light)
        return rgb[0]

    return train_loop(render, [logits], optimizer, gt_train, gt_test, train_poses, test_poses, iters,
                  snapshot_every, log_every, extra=lambda: {"texture": torch.sigmoid(logits).detach()})


def train_gaussians(gaussian_rasterizer, params, gt_train, gt_test, train_poses, test_poses, iters,
                    snapshot_every=None, log_every=100, seed=0):
    """
    Fits the raw Gaussian parameters (see initial_gaussians) in place.
    Returns a dict with the losses, the test PSNR (and its iterations) and test-view snapshots.
    """
    torch.manual_seed(seed)
    for p in params.values():
        p.requires_grad_(True)
    groups = [{"params": [params[k]], "lr": GAUSSIAN_LRS[k], "name": k} for k in params]
    optimizer = torch.optim.Adam(groups, eps=1e-15)

    def render(pose):
        rgb, _, _ = gaussian_rasterizer.rasterize(make_gaussians(params), pose)
        return rgb[0]

    return train_loop(render, list(params.values()), optimizer, gt_train, gt_test, train_poses, test_poses, iters,
                  snapshot_every, log_every)


def train_loop(render, tensors, optimizer, gt_train, gt_test, train_poses, test_poses, iters, snapshot_every,
               log_every, extra=None):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1.0 / iters))  # lr / 10 at the end

    def test_views():
        with torch.no_grad():
            return [render(p) for p in test_poses]

    def test_psnr(views):
        return float(np.mean([psnr(v, g) for v, g in zip(views, gt_test)]))

    views = test_views()
    hist = {"loss": [], "psnr_iters": [0], "psnr": [test_psnr(views)], "snapshots": [(0, views)]}
    t0 = time.time()
    for it in range(1, iters + 1):
        k = torch.randint(len(train_poses), (1,)).item()
        loss = (render(train_poses[k]) - gt_train[k]).abs().mean()
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
            print(f"iter {it:5d}  loss {np.mean(hist['loss'][-log_every:]):.4f}  ({time.time() - t0:.1f}s)")
    if extra is not None:
        hist.update(extra())
    return hist


def save_results(name, hist, gt_test, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    init_views, final_views = hist["snapshots"][0][1], hist["snapshots"][-1][1]
    rows = [np.concatenate([to_uint8(g), to_uint8(i), to_uint8(f)], axis=1)
            for g, i, f in zip(gt_test, init_views, final_views)]
    imageio.imwrite(os.path.join(out_dir, f"{name}_test_views_gt_init_trained.png"), np.concatenate(rows, axis=0))
    fig = Figure(figsize=(6, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(hist["loss"], linewidth=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("L1 loss")
    ax.set_yscale("log")
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_loss.png"), dpi=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=["texture", "gaussians", "all"], default="all")
    parser.add_argument("--iters", type=int, default=None, help="default: 1000 (texture), 3000 (gaussians)")
    parser.add_argument("--size", type=int, default=256, help="image width and height")
    parser.add_argument("--tex-size", type=int, default=512, help="resolution of the learned texture map")
    parser.add_argument("--texture-filter", choices=["mipmap", "bilinear"], default="mipmap",
                        help="texture sampling while learning the texture map")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=os.path.join(ROOT, "output", "textured_toy_training"))
    args = parser.parse_args()

    device = args.device
    mesh, light = load_scene(device)
    camera = OrbitCamera(image_width=args.size, image_height=args.size, fov_x=40, fov_y=40,
                         z_near=0.1, z_far=100, device=device)
    mesh_renderer = MeshRenderer(camera=camera, bg_color=(1.0, 1.0, 1.0), light=light)
    train_renderer = MeshRenderer(camera=camera, bg_color=(1.0, 1.0, 1.0), light=light,
                                  texture_filter=args.texture_filter)
    gaussian_rasterizer = GaussianRasterizer(camera=camera, bg_color=(1.0, 1.0, 1.0))
    train_poses, test_poses = orbit_poses(device)

    # pre-rendered images of the textured mesh
    with torch.no_grad():
        gt_train = [mesh_renderer.render(mesh, p)[0][0] for p in train_poses]
        gt_test = [mesh_renderer.render(mesh, p)[0][0] for p in test_poses]

    experiments = ["texture", "gaussians"] if args.experiment == "all" else [args.experiment]
    summary = []
    for name in experiments:
        print(f"\n=== experiment: {name} ===")
        if name == "texture":
            hist = train_texture(train_renderer, mesh, light, gt_train, gt_test, train_poses, test_poses,
                                 iters=args.iters or 1000, tex_size=args.tex_size, log_every=args.log_every)
            gt_small = F.interpolate(mesh.texture.permute(2, 0, 1)[None], size=(args.tex_size, args.tex_size),
                                     mode="area")[0].permute(1, 2, 0)
            os.makedirs(args.out, exist_ok=True)
            imageio.imwrite(os.path.join(args.out, "texture_gt_learned.png"),
                            np.concatenate([to_uint8(gt_small), to_uint8(hist["texture"])], axis=1))
        else:
            params = initial_gaussians(mesh, device)
            print(f"{params['means'].shape[0]} Gaussians")
            hist = train_gaussians(gaussian_rasterizer, params, gt_train, gt_test, train_poses, test_poses,
                                   iters=args.iters or 3000, log_every=args.log_every)
        save_results(name, hist, gt_test, args.out)
        summary.append((name, hist["psnr"][0], hist["psnr"][-1]))

    print("\n=== summary ===")
    for name, p0, p1 in summary:
        ok = p1 > p0 + 10.0
        print(f"{name:10s} test PSNR {p0:6.2f} -> {p1:6.2f} dB" + ("   [OK]" if ok else "   [CHECK]"))
    print(f"\nresults saved to {args.out}")


if __name__ == "__main__":
    main()
