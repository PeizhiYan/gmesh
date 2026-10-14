#
# Peizhi Yan
# Copyright (C) 2025
#

"""
Tests for the z-buffer (depth-limited) Gaussian rasterizer and the hybrid renderer.

Requires a CUDA GPU, gsplat v1.5.2 (and PyTorch3D + open3d for the renderer tests).
The first run JIT-compiles the CUDA extension, which takes a minute.

Run with:
    python -m pytest tests -v
or:
    python tests/test_zbuffer_rasterizer.py
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)

from gmesh.renderer.rasterizer_zbuffer import rasterize_to_pixels_zbuffer


DEVICE = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _random_scene(N=64, W=70, H=45, C=2, tile_size=16, channels=3, seed=0):
    """Random Gaussians seen by C cameras, projected and tile-intersected by gsplat."""
    g = torch.Generator().manual_seed(seed)
    means = (torch.rand(N, 3, generator=g) * 2 - 1) * torch.tensor([1.0, 0.7, 0.5]) + torch.tensor([0.0, 0.0, 3.0])
    quats = torch.randn(N, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, generator=g) * 0.15 + 0.03
    opacities = torch.rand(N, generator=g) * 0.5 + 0.05
    colors = torch.rand(C, N, channels, generator=g)

    viewmats = torch.eye(4).repeat(C, 1, 1)
    viewmats[1:, 0, 3] = 0.2  # shift the other cameras a bit
    Ks = torch.tensor([[60.0, 0.0, W / 2], [0.0, 60.0, H / 2], [0.0, 0.0, 1.0]]).repeat(C, 1, 1)

    to = lambda x: x.to(DEVICE)
    means, quats, scales, opacities, colors, viewmats, Ks = map(to, (means, quats, scales, opacities, colors, viewmats, Ks))

    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales, viewmats, Ks, W, H, packed=False, opacities=opacities
    )
    tile_width = math.ceil(W / tile_size)
    tile_height = math.ceil(H / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, packed=False, n_cameras=C
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    return dict(
        means2d=means2d.detach(), conics=conics.detach(), colors=colors,
        opacities=opacities.repeat(C, 1), depths=depths.detach(),
        isect_offsets=isect_offsets, flatten_ids=flatten_ids,
        W=W, H=H, C=C, N=N, tile_size=tile_size,
    )


def _leaves(scene):
    """Fresh leaf tensors (requiring grad) for the differentiable inputs."""
    return [scene[k].clone().requires_grad_(True) for k in ("means2d", "conics", "colors", "opacities")]


def _tile_membership(scene):
    """[C, H*W, N] bool: whether Gaussian n is in the intersection list of the pixel's tile."""
    C, N, W, H, ts = scene["C"], scene["N"], scene["W"], scene["H"], scene["tile_size"]
    offsets = scene["isect_offsets"].flatten().long()
    flatten_ids = scene["flatten_ids"].long()
    n_tiles_total = offsets.numel()
    ends = torch.cat([offsets[1:], offsets.new_tensor([flatten_ids.numel()])])
    tile_of_isect = torch.repeat_interleave(torch.arange(n_tiles_total, device=DEVICE), ends - offsets)
    member = torch.zeros(n_tiles_total, N, dtype=torch.bool, device=DEVICE)
    member[tile_of_isect, flatten_ids % N] = True
    member = member.view(C, -1, N)  # [C, n_tiles, N]

    tile_width = math.ceil(W / ts)
    ys, xs = torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing="ij")
    pix_tile = ((ys // ts) * tile_width + (xs // ts)).flatten()  # [H*W]
    return member[:, pix_tile, :]


def _reference(means2d, conics, colors, opacities, depths, depth_limits, member, W, H):
    """
    Brute-force PyTorch version of the depth-limited rasterizer (autograd for backward).
    Mirrors the CUDA kernel: front-to-back order, alpha threshold, 0.999 clamp,
    T <= 1e-4 termination, and the z-test.
    """
    C, N = means2d.shape[:2]
    ys, xs = torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing="ij")
    px = xs.flatten().float() + 0.5
    py = ys.flatten().float() + 0.5

    out_colors, out_alphas = [], []
    for c in range(C):
        zlim = depth_limits[c].flatten()
        T = torch.ones_like(px)
        done = torch.zeros_like(px, dtype=torch.bool)
        acc = torch.zeros(px.shape[0], colors.shape[-1], device=DEVICE)
        for n in torch.argsort(depths[c]).tolist():
            dx = means2d[c, n, 0] - px
            dy = means2d[c, n, 1] - py
            a, b, cc = conics[c, n, 0], conics[c, n, 1], conics[c, n, 2]
            sigma = 0.5 * (a * dx * dx + cc * dy * dy) + b * dx * dy
            alpha = torch.clamp_max(opacities[c, n] * torch.exp(-sigma), 0.999)
            valid = member[c, :, n] & (depths[c, n] < zlim) & (sigma >= 0) & (alpha >= 1.0 / 255.0) & ~done
            next_T = T * (1.0 - alpha)
            terminate = valid & (next_T <= 1e-4)
            done = done | terminate
            contribute = valid & ~terminate
            acc = acc + torch.where(contribute, alpha * T, torch.zeros_like(T))[:, None] * colors[c, n][None]
            T = torch.where(contribute, next_T, T)
        out_colors.append(acc.view(H, W, -1))
        out_alphas.append((1.0 - T).view(H, W, 1))
    return torch.stack(out_colors), torch.stack(out_alphas)


def _assert_close(actual, expected, name, rtol=1e-3, atol_scale=1e-4):
    atol = atol_scale * max(1.0, expected.abs().max().item())
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=lambda m: f"{name}: {m}")


def _random_weights(scene, channels):
    g = torch.Generator().manual_seed(123)
    C, H, W = scene["C"], scene["H"], scene["W"]
    w_c = torch.randn(C, H, W, channels, generator=g).to(DEVICE)
    w_a = torch.randn(C, H, W, 1, generator=g).to(DEVICE)
    return w_c, w_a


@pytest.mark.parametrize("channels", [3, 4, 6])
def test_infinite_depth_limits_match_gsplat(channels):
    """With no occluder, the result (and gradients) must equal gsplat's rasterizer."""
    scene = _random_scene(channels=channels)
    W, H, C, ts = scene["W"], scene["H"], scene["C"], scene["tile_size"]
    limits = torch.full((C, H, W), float("inf"), device=DEVICE)
    w_c, w_a = _random_weights(scene, channels)

    ref_in = _leaves(scene)
    ref_c, ref_a = rasterize_to_pixels(
        *ref_in, W, H, ts, scene["isect_offsets"], scene["flatten_ids"], packed=False
    )
    ((ref_c * w_c).sum() + (ref_a * w_a).sum()).backward()

    our_in = _leaves(scene)
    our_c, our_a = rasterize_to_pixels_zbuffer(
        *our_in, scene["depths"], limits, W, H, ts, scene["isect_offsets"], scene["flatten_ids"]
    )
    ((our_c * w_c).sum() + (our_a * w_a).sum()).backward()

    _assert_close(our_c, ref_c, "colors", atol_scale=1e-6)
    _assert_close(our_a, ref_a, "alphas", atol_scale=1e-6)
    for name, ours, ref in zip(("means2d", "conics", "colors", "opacities"), our_in, ref_in):
        _assert_close(ours.grad, ref.grad, f"grad {name}", atol_scale=1e-5)


@pytest.mark.parametrize("channels", [3, 4])
def test_depth_limits_match_reference(channels):
    """With an occluder, compare forward and gradients to a brute-force reference."""
    scene = _random_scene(channels=channels, seed=1)
    W, H, C, ts = scene["W"], scene["H"], scene["C"], scene["tile_size"]

    # a slanted occluder through the Gaussians, nothing in some pixels, and a
    # region in the very first tile where every Gaussian is occluded
    xs = torch.arange(W, device=DEVICE).float()
    limits = (2.6 + 0.8 * xs / W).view(1, 1, W).repeat(C, H, 1)
    limits[:, H // 2:, : W // 3] = float("inf")
    limits[:, :8, :8] = 0.5

    w_c, w_a = _random_weights(scene, channels)
    member = _tile_membership(scene)

    ref_in = _leaves(scene)
    ref_c, ref_a = _reference(*ref_in, scene["depths"], limits, member, W, H)
    ((ref_c * w_c).sum() + (ref_a * w_a).sum()).backward()

    our_in = _leaves(scene)
    our_c, our_a = rasterize_to_pixels_zbuffer(
        *our_in, scene["depths"], limits, W, H, ts, scene["isect_offsets"], scene["flatten_ids"]
    )
    ((our_c * w_c).sum() + (our_a * w_a).sum()).backward()

    # the occluder actually removed something
    assert (our_a.sum() < rasterize_to_pixels(*_leaves(scene), W, H, ts, scene["isect_offsets"],
                                              scene["flatten_ids"], packed=False)[1].sum()).item()
    # fully occluded pixels are empty
    assert our_a[:, :8, :8].abs().max().item() == 0.0

    _assert_close(our_c, ref_c, "colors")
    _assert_close(our_a, ref_a, "alphas")
    for name, ours, ref in zip(("means2d", "conics", "colors", "opacities"), our_in, ref_in):
        _assert_close(ours.grad, ref.grad, f"grad {name}")


def test_depth_bias():
    """depth_bias shifts the depth limit."""
    scene = _random_scene(seed=2)
    W, H, C, ts = scene["W"], scene["H"], scene["C"], scene["tile_size"]
    args = (scene["means2d"], scene["conics"], scene["colors"], scene["opacities"], scene["depths"])
    limits = torch.full((C, H, W), 3.0, device=DEVICE)
    c1, a1 = rasterize_to_pixels_zbuffer(*args, limits + 0.25, W, H, ts, scene["isect_offsets"], scene["flatten_ids"])
    c2, a2 = rasterize_to_pixels_zbuffer(*args, limits, W, H, ts, scene["isect_offsets"], scene["flatten_ids"],
                                         depth_bias=0.25)
    torch.testing.assert_close(c1, c2)
    torch.testing.assert_close(a1, a2)


# ---------------------------------------------------------------------------
# Hybrid renderer tests (need PyTorch3D and open3d)
# ---------------------------------------------------------------------------

def _load_spot():
    pytest.importorskip("pytorch3d")
    o3d_utils = pytest.importorskip("gmesh.utils.o3d_utils")
    import numpy as np
    vertices, faces = o3d_utils._read_obj_file(os.path.join(ROOT, "assets", "data", "spot.obj"), uv=False)
    theta = np.pi * 0.8
    rotation_matrix = torch.tensor([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ], dtype=torch.float32, device=DEVICE)
    vertices = torch.from_numpy(vertices).float().to(DEVICE) @ rotation_matrix.T
    faces = torch.from_numpy(faces).to(DEVICE)
    return vertices, faces


@pytest.mark.parametrize("width,height", [(320, 160), (160, 320)])
def test_mesh_is_pixel_aligned_with_gaussians(width, height):
    """
    Facet centers lie on the mesh surface, so when projected by gsplat they must
    land on mesh pixels, and they can never be in front of the mesh z-buffer.
    Checks the non-square NDC handling and the depth conversion of the mesh renderer.
    """
    from gmesh.scenes.mesh import Mesh
    from gmesh.scenes.cameras import PerspectiveCamera
    from gmesh.renderer.renderer_pytorch3d import CustomMeshRenderer
    from gmesh.utils.mesh2gaussians_utils import mesh_to_gaussians

    vertices, faces = _load_spot()
    mesh = Mesh(verts=vertices, faces=faces)
    gaussians = mesh_to_gaussians(mesh=mesh, method="facet")

    camera = PerspectiveCamera(image_width=width, image_height=height, fov_x=40,
                               fov_y=40 * height / width, z_near=0.1, z_far=100, device=DEVICE)
    camera_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 7.0]], device=DEVICE)

    _, depth_m, alpha_m = CustomMeshRenderer(camera=camera).render(mesh=mesh, camera_pose=camera_pose)

    viewmats = camera.get_view_matrix(camera_pose)
    Ks = camera.get_intrinsic_matrix(batch_size=1)
    pts_cam = gaussians.means @ viewmats[0, :3, :3].T + viewmats[0, :3, 3]  # [F, 3]
    uv = pts_cam[:, :2] / pts_cam[:, 2:3] * Ks[0, [0, 1], [0, 1]] + Ks[0, :2, 2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    assert inside.float().mean().item() > 0.99
    u = uv[inside, 0].long()
    v = uv[inside, 1].long()

    z_center = pts_cam[inside, 2]

    # 3x3 neighborhoods, to tolerate pixel quantization at silhouettes
    covered = alpha_m[..., 0]  # [1, H, W]
    covered_n = F.max_pool2d(covered[None], 3, stride=1, padding=1)[0, 0]  # [H, W]
    depth_inf = torch.where(covered > 0, depth_m[..., 0], torch.full_like(covered, float("inf")))
    depth_min_n = -F.max_pool2d(-depth_inf[None], 3, stride=1, padding=1)[0, 0]  # [H, W]

    on_mesh = covered_n[v, u] > 0
    assert on_mesh.float().mean().item() > 0.99, on_mesh.float().mean().item()

    # nothing is in front of the nearest surface
    not_in_front = depth_min_n[v, u] <= z_center * 1.02
    assert not_in_front.float().mean().item() > 0.99, not_in_front.float().mean().item()

    # and the visible facets are on the z-buffer surface
    exact = covered[0, v, u] > 0
    on_surface = (depth_m[0, v, u, 0] - z_center).abs() <= z_center * 0.01
    assert (on_surface & exact).float().mean().item() > 0.2, (on_surface & exact).float().mean().item()


def test_hybrid_renderer_occlusion_and_gradients():
    from gmesh.scenes.mesh import Mesh
    from gmesh.scenes.cameras import PerspectiveCamera
    from gmesh.renderer.gmesh_renderer import GmeshRenderer
    from gmesh.renderer.shader import simple_phone_shading
    from gmesh.utils.mesh2gaussians_utils import mesh_to_gaussians

    vertices, faces = _load_spot()
    mesh = Mesh(verts=vertices, faces=faces)
    mesh.colors = simple_phone_shading(vertices=vertices, faces=faces)
    camera = PerspectiveCamera(image_width=256, image_height=128, fov_x=40, fov_y=20,
                               z_near=0.1, z_far=100, device=DEVICE)
    camera_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 7.0]], device=DEVICE)
    renderer = GmeshRenderer(camera=camera, tile_size=16, bg_color=(1.0, 1.0, 1.0))

    # Gaussians pushed far behind the mesh: they must be invisible wherever the mesh is
    gaussians = mesh_to_gaussians(mesh=Mesh(verts=vertices.clone(), faces=faces), method="facet3", trainable=True)
    with torch.no_grad():
        gaussians.means[:, 2] -= 3.0
    ret = renderer.render(mesh=mesh, gaussians=gaussians, camera_pose=camera_pose, return_all=True)
    rgb_m, _, alpha_m = ret["mesh"]
    rgb, depth, alpha = ret["blend"]
    covered = alpha_m[..., 0] > 0
    torch.testing.assert_close(rgb[covered], rgb_m[covered])
    assert torch.isfinite(rgb).all() and torch.isfinite(depth).all() and torch.isfinite(alpha).all()

    # Gaussians in front of the mesh change the image, and gradients flow
    gaussians = mesh_to_gaussians(mesh=Mesh(verts=vertices.clone(), faces=faces), method="facet3", trainable=True)
    with torch.no_grad():
        gaussians.means[:, 2] += 1.0
    rgb, depth, alpha = renderer.render(mesh=mesh, gaussians=gaussians, camera_pose=camera_pose)
    assert (rgb[covered] - rgb_m[covered]).abs().max().item() > 0.1
    rgb.mean().backward()
    for name in ("means", "scales", "opacities", "colors"):
        grad = getattr(gaussians, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum().item() > 0, name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
