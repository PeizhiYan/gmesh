#
# Peizhi Yan
# Copyright (C) 2026
#

"""
Texture learning example: recover the texture map of a mesh from multi-view
images, through the differentiable mesh rasterizer.

Ground truth: the textured Spot mesh (assets/data/spot), rendered from 24
orbit views (training) and 6 other views (testing).

Starting from a uniform gray texture, Adam optimizes the texture map (through a
sigmoid, so it stays in [0, 1]) with an L1 photometric loss. The geometry and
texture coordinates are fixed.

Usage (from the repository root):
    python examples/train_texture.py
    python examples/train_texture.py --filter bilinear --iters 500 --size 256

It prints the training loss and the test PSNR, and saves to --out: test views
(ground truth | initialization | trained), the learned texture next to the
ground-truth one, and the loss curve.
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
from gmesh.scenes.cameras import OrbitCamera
from gmesh.renderer.rasterizer_mesh import MeshRenderer
from gmesh.utils.mesh_io import load_obj, load_texture

SPOT_DIR = os.path.join(ROOT, "assets", "data", "spot")


def load_spot(device):
    verts, faces, uvs, faces_uv = load_obj(os.path.join(SPOT_DIR, "spot_triangulated.obj"), device=device)
    theta = math.pi * 0.8
    rotation = torch.tensor([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)],
    ], dtype=torch.float32, device=device)
    texture = load_texture(os.path.join(SPOT_DIR, "spot_texture.png"), device=device)
    return verts @ rotation.T, faces, uvs, faces_uv, texture


def orbit_poses(device):
    train = [torch.tensor([[(0.45 if k % 2 else -0.3), 2 * math.pi * k / 24, 3.0]], device=device) for k in range(24)]
    test = [torch.tensor([[0.15, 2 * math.pi * (k + 0.5) / 6, 3.0]], device=device) for k in range(6)]
    return train, test


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).clamp(min=1e-10)
    return (-10.0 * torch.log10(mse)).item()


def to_uint8(img):
    return (img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--size", type=int, default=256, help="image width and height")
    parser.add_argument("--tex-size", type=int, default=512, help="resolution of the learned texture map")
    parser.add_argument("--filter", choices=["mipmap", "bilinear"], default="mipmap", help="texture filter")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=os.path.join(ROOT, "output", "texture_training"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = args.device
    torch.manual_seed(0)
    verts, faces, uvs, faces_uv, gt_texture = load_spot(device)
    camera = OrbitCamera(image_width=args.size, image_height=args.size, fov_x=40, fov_y=40,
                         z_near=0.1, z_far=100, device=device)
    renderer = MeshRenderer(camera=camera, bg_color=(1.0, 1.0, 1.0), antialias=True, texture_filter=args.filter)
    train_poses, test_poses = orbit_poses(device)

    def render(texture, pose):
        mesh = Mesh(verts=verts, faces=faces, uvs=uvs, faces_uv=faces_uv, texture=texture)
        rgb, _, _ = renderer.render(mesh, pose)
        return rgb[0]

    with torch.no_grad():
        gt_train = [render(gt_texture, p) for p in train_poses]
        gt_test = [render(gt_texture, p) for p in test_poses]

    # learned texture: uniform gray to start with
    logits = torch.zeros(args.tex_size, args.tex_size, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=args.lr)

    def test_psnr():
        with torch.no_grad():
            texture = torch.sigmoid(logits)
            preds = [render(texture, p) for p in test_poses]
        return float(np.mean([psnr(p, g) for p, g in zip(preds, gt_test)])), preds

    psnr_init, init_views = test_psnr()
    print(f"test PSNR at initialization: {psnr_init:.2f} dB")

    losses = []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        k = torch.randint(len(train_poses), (1,)).item()
        pred = render(torch.sigmoid(logits), train_poses[k])
        loss = (pred - gt_train[k]).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if it % args.log_every == 0 or it == args.iters:
            print(f"iter {it:5d}  loss {np.mean(losses[-args.log_every:]):.4f}  ({time.time() - t0:.1f}s)")

    psnr_final, final_views = test_psnr()
    print(f"test PSNR: {psnr_init:.2f} -> {psnr_final:.2f} dB")

    # outputs
    rows = [np.concatenate([to_uint8(g), to_uint8(i), to_uint8(f)], axis=1)
            for g, i, f in zip(gt_test, init_views, final_views)]
    imageio.imwrite(os.path.join(args.out, "test_views_gt_init_trained.png"), np.concatenate(rows, axis=0))
    gt_small = F.interpolate(gt_texture.permute(2, 0, 1)[None], size=(args.tex_size, args.tex_size),
                             mode="area")[0].permute(1, 2, 0)
    imageio.imwrite(os.path.join(args.out, "texture_gt_learned.png"),
                    np.concatenate([to_uint8(gt_small), to_uint8(torch.sigmoid(logits))], axis=1))
    fig = Figure(figsize=(6, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(losses, linewidth=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("L1 loss")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "loss.png"), dpi=120)
    print(f"results saved to {args.out}")


if __name__ == "__main__":
    main()
