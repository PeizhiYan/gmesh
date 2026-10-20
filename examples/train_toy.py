#
# Peizhi Yan
# Copyright (C) 2026
#

"""
Toy training example: fit 3D Gaussians and/or a mesh through the
differentiable hybrid renderer, from multi-view images.

Ground truth: the example mesh (assets/data/spot.obj, shaded vertex colors)
and two colorful rings of Gaussians that pass *through* the mesh (parts are in
front of it, behind it, and crossing its surface). It is rendered from 24
orbit views (training) and 6 other views (testing).

Starting from perturbed parameters, Adam optimizes, with an L1 photometric loss:

  gaussians: the Gaussians (means, rotations, scales, opacities, colors);
             the mesh is fixed and occludes them
  mesh:      a translation of the mesh and its vertex colors; the Gaussians
             are fixed (gradients through silhouettes, the mesh depth vs. the
             Gaussians, and the color interpolation)
  joint:     both at once
  all:       the three experiments, one after the other (default)

Usage (from the repository root):
    python examples/train_toy.py
    python examples/train_toy.py --experiment mesh --iters 600 --size 256

It prints the training loss and the test PSNR, and saves to --out, for each
experiment, test views (ground truth | initialization | trained) and the
loss curve.
"""

import argparse
import colorsys
import math
import os
import sys
import time

import imageio.v2 as imageio
import numpy as np
from matplotlib.figure import Figure  # no pyplot: importing this file must not change the plotting backend
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gmesh.utils.o3d_utils as o3d_utils
from gmesh.scenes.mesh import Mesh
from gmesh.scenes.gaussians import Gaussians
from gmesh.scenes.cameras import OrbitCamera
from gmesh.renderer.gmesh_renderer import GmeshRenderer
from gmesh.renderer.mesh_shader import simple_phone_shading

SH_C0 = 0.28209479177387814  # rgb = 0.5 + SH_C0 * sh (degree 0)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def load_mesh(device):
    vertices, faces = o3d_utils._read_obj_file(os.path.join(ROOT, "assets", "data", "spot.obj"), uv=False)
    theta = np.pi * 0.8
    rotation = torch.tensor([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ], dtype=torch.float32, device=device)
    vertices = torch.from_numpy(vertices).float().to(device) @ rotation.T
    faces = torch.from_numpy(faces).to(device)
    colors = simple_phone_shading(vertices=vertices, faces=faces,
                                  ambient_color=torch.tensor([0.05, 0.15, 0.35], device=device),
                                  diffuse_color=torch.tensor([0.10, 0.35, 0.75], device=device))
    return vertices, faces, colors.clamp(0.02, 0.98)


def ground_truth_gaussians(n_per_ring, device, seed=0):
    """Two rings of Gaussians (around the y and z axes) that cross the mesh. Returns raw parameters."""
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(2, n_per_ring, generator=g) * 2 * math.pi
    r = 0.85 + 0.04 * torch.randn(2, n_per_ring, generator=g)
    wobble = 0.05 * torch.sin(3 * t)
    ring_y = torch.stack([r[0] * torch.cos(t[0]), wobble[0], r[0] * torch.sin(t[0])], dim=-1)  # horizontal
    ring_z = torch.stack([r[1] * torch.cos(t[1]), r[1] * torch.sin(t[1]), wobble[1]], dim=-1)  # vertical
    means = torch.cat([ring_y, ring_z]) + 0.02 * torch.randn(2 * n_per_ring, 3, generator=g)

    hue = torch.cat([t[0], t[1] + math.pi]) / (2 * math.pi) % 1.0
    rgb = torch.tensor([colorsys.hsv_to_rgb(h, 0.85, 0.95) for h in hue.tolist()])
    n = means.shape[0]
    return {
        "means": means.to(device),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device),
        "log_scales": torch.log(0.02 + 0.02 * torch.rand(n, 3, generator=g)).to(device),
        "opacity_logits": torch.full((n,), 2.5).to(device),  # opacity ~0.92
        "sh": ((rgb - 0.5) / SH_C0)[:, None, :].to(device),  # [N, 1, 3]
    }


def perturbed_gaussians(gt, device, seed=1):
    """Initialization for training: noisy positions, random rotations/colors, default scales/opacities."""
    g = torch.Generator().manual_seed(seed)
    n = gt["means"].shape[0]
    return {
        "means": gt["means"] + (0.08 * torch.randn(n, 3, generator=g)).to(device),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device),
        "log_scales": torch.full((n, 3), math.log(0.03)).to(device),
        "opacity_logits": torch.zeros(n).to(device),  # opacity 0.5
        "sh": (0.5 * torch.randn(n, 1, 3, generator=g)).to(device),
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


def make_mesh(vertices, faces, mesh_params):
    return Mesh(verts=vertices + mesh_params["translation"], faces=faces,
                colors=torch.sigmoid(mesh_params["color_logits"]))


def orbit_poses(device):
    train = [torch.tensor([[(0.35 if k % 2 else -0.25), 2 * math.pi * k / 24, 5.0]], device=device) for k in range(24)]
    test = [torch.tensor([[0.15, 2 * math.pi * (k + 0.5) / 6, 5.0]], device=device) for k in range(6)]
    return train, test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def psnr(a, b):
    mse = torch.mean((a - b) ** 2).clamp(min=1e-10)
    return (-10.0 * torch.log10(mse)).item()


def render_rgb(renderer, mesh, gaussians, pose):
    """[H, W, 3] image of a single view"""
    rgb, _, _ = renderer.render(mesh=mesh, gaussians=gaussians, camera_pose=pose)
    return rgb[0]


def to_uint8(img):
    return (img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def run_experiment(name, args, renderer, vertices, faces, gt_mesh_colors, gt_gaussian_params, train_poses,
                   test_poses, device):
    print(f"\n=== experiment: {name} ===")
    train_gaussians = name in ("gaussians", "joint")
    train_mesh = name in ("mesh", "joint")

    gt_mesh_params = {
        "translation": torch.zeros(3, device=device),
        "color_logits": torch.logit(gt_mesh_colors),
    }

    # ground-truth images
    with torch.no_grad():
        gt_mesh = make_mesh(vertices, faces, gt_mesh_params)
        gt_gaussians = make_gaussians(gt_gaussian_params)
        gt_train = [render_rgb(renderer, gt_mesh, gt_gaussians, pose) for pose in train_poses]
        gt_test = [render_rgb(renderer, gt_mesh, gt_gaussians, pose) for pose in test_poses]

    # parameters to optimize (and fixed ones)
    if train_gaussians:
        g_params = {k: v.clone().requires_grad_(True) for k, v in perturbed_gaussians(gt_gaussian_params, device).items()}
    else:
        g_params = {k: v.clone() for k, v in gt_gaussian_params.items()}
    if train_mesh:
        m_params = {
            "translation": torch.tensor([0.12, -0.08, 0.10], device=device).requires_grad_(True),
            "color_logits": torch.zeros_like(gt_mesh_colors).requires_grad_(True),  # gray
        }
    else:
        m_params = {k: v.clone() for k, v in gt_mesh_params.items()}

    lrs = {
        "means": 2e-3, "quats": 1e-2, "log_scales": 1e-2, "opacity_logits": 5e-2, "sh": 2e-2,
        "translation": 2e-3, "color_logits": 5e-2,
    }
    groups = []
    if train_gaussians:
        groups += [{"params": [g_params[k]], "lr": lrs[k], "name": k} for k in g_params]
    if train_mesh:
        groups += [{"params": [m_params[k]], "lr": lrs[k], "name": k} for k in m_params]
    optimizer = torch.optim.Adam(groups, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1.0 / args.iters))

    def render_all(poses):
        with torch.no_grad():
            mesh, gaussians = make_mesh(vertices, faces, m_params), make_gaussians(g_params)
            return [render_rgb(renderer, mesh, gaussians, pose) for pose in poses]

    init_test = render_all(test_poses)
    psnr_init = float(np.mean([psnr(a, b) for a, b in zip(init_test, gt_test)]))
    t_err_init = m_params["translation"].detach().norm().item()
    print(f"init:  test PSNR {psnr_init:6.2f} dB" + (f" | mesh translation error {t_err_init:.4f}" if train_mesh else ""))

    losses = []
    g = torch.Generator().manual_seed(2)
    start = time.time()
    for it in range(1, args.iters + 1):
        k = int(torch.randint(len(train_poses), (1,), generator=g))
        mesh, gaussians = make_mesh(vertices, faces, m_params), make_gaussians(g_params)
        rgb = render_rgb(renderer, mesh, gaussians, train_poses[k])
        loss = (rgb - gt_train[k]).abs().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if train_gaussians:
            with torch.no_grad():
                g_params["quats"].data = torch.nn.functional.normalize(g_params["quats"].data, dim=-1)
        losses.append(loss.item())

        if it % args.log_every == 0 or it == args.iters:
            msg = f"iter {it:5d} | loss {np.mean(losses[-args.log_every:]):.4f}"
            if train_mesh:
                msg += f" | mesh translation error {m_params['translation'].detach().norm().item():.4f}"
            print(msg + f" | {time.time() - start:.1f}s")

    final_test = render_all(test_poses)
    psnr_final = float(np.mean([psnr(a, b) for a, b in zip(final_test, gt_test)]))
    t_err_final = m_params["translation"].detach().norm().item()
    print(f"final: test PSNR {psnr_final:6.2f} dB" + (f" | mesh translation error {t_err_final:.4f}" if train_mesh else ""))

    # outputs: test views (rows) x [ground truth | initialization | trained], and the loss curve
    os.makedirs(args.out, exist_ok=True)
    rows = [np.concatenate([to_uint8(a), to_uint8(b), to_uint8(c)], axis=1)
            for a, b, c in zip(gt_test[:3], init_test[:3], final_test[:3])]
    imageio.imwrite(os.path.join(args.out, f"{name}_test_views.png"), np.concatenate(rows, axis=0))
    fig = Figure(figsize=(6, 3.5))
    ax = fig.add_subplot()
    ax.plot(losses, linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("L1 loss")
    ax.set_title(f"toy training: {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{name}_loss.png"), dpi=120)

    ok = psnr_final > psnr_init + 3.0
    if train_mesh:
        ok = ok and t_err_final < 0.2 * t_err_init
    return {"experiment": name, "psnr_init": psnr_init, "psnr_final": psnr_final,
            "t_err_init": t_err_init if train_mesh else None,
            "t_err_final": t_err_final if train_mesh else None, "ok": ok}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=["gaussians", "mesh", "joint", "all"], default="all")
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--size", type=int, default=256, help="image width and height")
    parser.add_argument("--gaussians-per-ring", type=int, default=1500)
    parser.add_argument("--softness", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out", default=os.path.join(ROOT, "output", "toy_training"))
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    vertices, faces, gt_mesh_colors = load_mesh(device)
    gt_gaussian_params = ground_truth_gaussians(args.gaussians_per_ring, device)
    camera = OrbitCamera(image_width=args.size, image_height=args.size, fov_x=40, fov_y=40,
                         z_near=0.1, z_far=100, device=device)
    renderer = GmeshRenderer(camera=camera, tile_size=16, bg_color=(1.0, 1.0, 1.0), softness=args.softness)
    train_poses, test_poses = orbit_poses(device)

    experiments = ["gaussians", "mesh", "joint"] if args.experiment == "all" else [args.experiment]
    results = [run_experiment(name, args, renderer, vertices, faces, gt_mesh_colors, gt_gaussian_params,
                              train_poses, test_poses, device) for name in experiments]

    print("\n=== summary ===")
    for r in results:
        line = f"{r['experiment']:10s} test PSNR {r['psnr_init']:6.2f} -> {r['psnr_final']:6.2f} dB"
        if r["t_err_init"] is not None:
            line += f" | mesh translation error {r['t_err_init']:.4f} -> {r['t_err_final']:.4f}"
        print(line + ("   [OK]" if r["ok"] else "   [CHECK]"))
    print(f"\nimages and loss curves saved to {args.out}")


if __name__ == "__main__":
    main()
