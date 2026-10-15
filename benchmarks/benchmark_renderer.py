#
# Peizhi Yan
# Copyright (C) 2025
#

"""
Benchmark the Gmesh renderers.

Scene: the example mesh (assets/data/spot.obj) with Gaussians on its surface
('facet3'), plus optional random Gaussians around it (some of them hidden
behind the mesh, where occlusion culling helps).

Usage (from the repository root):
    python benchmarks/benchmark_renderer.py
    python benchmarks/benchmark_renderer.py --width 1920 --height 1080 --extra-gaussians 500000 --backward

Reports the average time per frame (CUDA events) of each pipeline, forward only
and, with --backward, forward + backward. For reference, it also times gsplat
alone and the mesh and Gaussians rendered separately (without any occlusion
between them).
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gmesh.utils.o3d_utils as o3d_utils
from gmesh.scenes.mesh import Mesh
from gmesh.scenes.gaussians import Gaussians
from gmesh.scenes.cameras import PerspectiveCamera
from gmesh.renderer.gmesh_renderer import GmeshRenderer
from gmesh.renderer.rasterizer_mesh import MeshRenderer
from gmesh.renderer.rasterizer_gsplat import GaussianRasterizer
from gmesh.renderer.shader import simple_phone_shading
from gmesh.utils.mesh2gaussians_utils import mesh_to_gaussians


def build_scene(args, device):
    vertices, faces = o3d_utils._read_obj_file(os.path.join(ROOT, "assets", "data", "spot.obj"), uv=False)
    theta = np.pi * 0.8
    rotation = torch.tensor([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ], dtype=torch.float32, device=device)
    vertices = torch.from_numpy(vertices).float().to(device) @ rotation.T
    faces = torch.from_numpy(faces).to(device)

    surface = mesh_to_gaussians(mesh=Mesh(verts=vertices.clone(), faces=faces), method="facet3")
    means, quats, scales = surface.means, surface.quats, surface.scales
    opacities, colors = surface.opacities, surface.colors
    colors[..., 0], colors[..., 1], colors[..., 2] = 1.0, 1.0, 0.2

    if args.extra_gaussians > 0:
        g = torch.Generator(device="cpu").manual_seed(0)
        n = args.extra_gaussians
        extra_means = ((torch.rand(n, 3, generator=g) * 2 - 1) * 1.5).to(device)
        extra_quats = torch.randn(n, 4, generator=g).to(device)
        extra_scales = (torch.rand(n, 3, generator=g) * 0.02 + 0.005).to(device)
        extra_opacities = (torch.rand(n, generator=g) * 0.9 + 0.1).to(device)
        extra_colors = torch.rand(n, 1, 3, generator=g).to(device)
        means = torch.cat([means, extra_means])
        quats = torch.cat([quats, extra_quats])
        scales = torch.cat([scales, extra_scales])
        opacities = torch.cat([opacities, extra_opacities])
        colors = torch.cat([colors, extra_colors])

    return vertices, faces, (means, quats, scales, opacities, colors)


def make_inputs(vertices, faces, gaussian_params, requires_grad):
    mesh = Mesh(verts=vertices.clone().requires_grad_(requires_grad), faces=faces)
    mesh.colors = simple_phone_shading(vertices=vertices, faces=faces).detach().requires_grad_(requires_grad)
    params = [p.clone().requires_grad_(requires_grad) for p in gaussian_params]
    gaussians = Gaussians(*params)
    leaves = [mesh.verts, mesh.colors] + params
    return mesh, gaussians, leaves


def time_ms(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--extra-gaussians", type=int, default=200_000,
                        help="random Gaussians around the mesh, in addition to the ones on its surface")
    parser.add_argument("--backward", action="store_true", help="also time forward + backward")
    args = parser.parse_args()

    device = "cuda"
    torch.backends.cudnn.benchmark = True
    vertices, faces, gaussian_params = build_scene(args, device)
    camera = PerspectiveCamera(image_width=args.width, image_height=args.height, fov_x=40,
                               fov_y=40.0 * args.height / args.width, z_near=0.1, z_far=100, device=device)
    camera_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 7.0]], device=device)
    bg = (1.0, 1.0, 1.0)

    mesh_aa = MeshRenderer(camera=camera, bg_color=bg, antialias=True)
    mesh_no_aa = MeshRenderer(camera=camera, bg_color=bg, antialias=False)
    gsplat_rasterizer = GaussianRasterizer(camera=camera, bg_color=bg)
    hybrid = GmeshRenderer(camera=camera, bg_color=bg, antialias=True, cull_occluded=True)
    hybrid_no_cull = GmeshRenderer(camera=camera, bg_color=bg, antialias=True, cull_occluded=False)
    hybrid_no_aa = GmeshRenderer(camera=camera, bg_color=bg, antialias=False, cull_occluded=True)

    def separate(mesh, gaussians):
        # mesh and Gaussians rendered on their own, without any occlusion between them
        rgb_m, depth_m, alpha_m = mesh_aa.render(mesh=mesh, camera_pose=camera_pose)
        rgb_g, depth_g, alpha_g = gsplat_rasterizer.rasterize(gaussians=gaussians, camera_pose=camera_pose)
        return rgb_m + rgb_g, depth_m + depth_g, alpha_m + alpha_g

    cases = [
        ("mesh only", "Gmesh mesh rasterizer (antialiased)", lambda m, g: mesh_aa.render(mesh=m, camera_pose=camera_pose)),
        ("mesh only", "Gmesh mesh rasterizer (no AA)", lambda m, g: mesh_no_aa.render(mesh=m, camera_pose=camera_pose)),
        ("3DGS only", "gsplat", lambda m, g: gsplat_rasterizer.rasterize(gaussians=g, camera_pose=camera_pose)),
        ("hybrid", "GmeshRenderer (AA + culling, default)", lambda m, g: hybrid.render(mesh=m, gaussians=g, camera_pose=camera_pose)),
        ("hybrid", "GmeshRenderer (AA, no culling)", lambda m, g: hybrid_no_cull.render(mesh=m, gaussians=g, camera_pose=camera_pose)),
        ("hybrid", "GmeshRenderer (no AA, culling)", lambda m, g: hybrid_no_aa.render(mesh=m, gaussians=g, camera_pose=camera_pose)),
        ("reference", "mesh + gsplat, separately (no occlusion)", separate),
    ]

    n_gaussians = gaussian_params[0].shape[0]
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"image {args.width}x{args.height}, mesh {faces.shape[0]} faces, {n_gaussians} Gaussians, "
          f"{args.iters} iterations (+{args.warmup} warmup)")

    # how much occlusion culling removes in this scene
    with torch.no_grad():
        mesh, gaussians, _ = make_inputs(vertices, faces, gaussian_params, requires_grad=False)
        raster = hybrid.mesh_renderer.rasterize(mesh=mesh, camera_pose=camera_pose)
        depth_limits = torch.where(raster["alpha"] > 0, raster["depth"], torch.full_like(raster["depth"], float("inf")))
        layers = hybrid.gaussian_rasterizer.rasterize_hybrid(
            gaussians=gaussians, camera_pose=camera_pose, depth_limits=depth_limits[..., 0],
            coverages=raster["alpha"][..., 0], cull_occluded=True,
        )
        print(f"occlusion culling: {layers['n_isects_rendered']} of {layers['n_isects']} tile intersections rendered "
              f"({100.0 * (1 - layers['n_isects_rendered'] / max(1, layers['n_isects'])):.1f}% culled)")
    print()

    header = f"{'':10s} {'pipeline':42s} {'forward (ms)':>13s}"
    if args.backward:
        header += f" {'fwd+bwd (ms)':>13s}"
    print(header)
    print("-" * len(header))
    for group, name, fn in cases:
        mesh, gaussians, leaves = make_inputs(vertices, faces, gaussian_params, requires_grad=False)
        with torch.no_grad():
            fwd = time_ms(lambda: fn(mesh, gaussians), args.iters, args.warmup)
        row = f"{group:10s} {name:42s} {fwd:13.2f}"
        if args.backward:
            mesh, gaussians, leaves = make_inputs(vertices, faces, gaussian_params, requires_grad=True)

            def step():
                out = fn(mesh, gaussians)
                loss = out[0].mean() + out[2].mean()
                loss.backward()
                for leaf in leaves:
                    leaf.grad = None

            row += f" {time_ms(step, args.iters, args.warmup):13.2f}"
        print(row)


if __name__ == "__main__":
    main()
