#
# Peizhi Yan
# Copyright (C) 2025
#

"""
Tests for the mesh rasterizer and the hybrid renderer (GmeshRenderer).

Requires a CUDA GPU, gsplat v1.5.2 and open3d (to read the example mesh).
PyTorch3D is optional: if installed, the mesh rasterizer is also compared to the
former PyTorch3D-based renderer.

Run with:
    python -m pytest tests -v
"""

import math

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from gmesh.renderer.rasterizer_mesh import MeshRenderer, project_vertices, rasterize_mesh

from common import DEVICE, load_spot, pixel_centers


def _camera(W, H, dtype=torch.float32):
    viewmats = torch.eye(4, dtype=dtype, device=DEVICE)[None]
    Ks = torch.tensor([[50.0, 0.0, W / 2], [0.0, 50.0, H / 2], [0.0, 0.0, 1.0]], dtype=dtype, device=DEVICE)[None]
    return viewmats, Ks


def _random_mesh(F_=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = (torch.rand(F_, 1, 3, generator=g) * 2 - 1) * torch.tensor([0.8, 0.5, 0.6]) + torch.tensor([0.0, 0.0, 3.0])
    offsets = torch.randn(F_, 3, 3, generator=g) * 0.3
    verts = (centers + offsets).reshape(-1, 3)
    faces = torch.arange(F_ * 3).view(F_, 3)
    return verts.to(DEVICE), faces.to(DEVICE)


def _mesh_reference(verts, faces, viewmats, Ks, W, H, near, far):
    """Brute force: every face against every pixel."""
    verts_cam, uv = project_vertices(verts, viewmats, Ks)
    px, py = pixel_centers(W, H)
    best_iz = torch.zeros(W * H, device=DEVICE)
    best_id = torch.full((W * H,), -1, dtype=torch.long, device=DEVICE)
    tri_uv = uv[0][faces]               # [F, 3, 2]
    tri_z = verts_cam[0][faces][..., 2]  # [F, 3]
    for f in range(faces.shape[0]):
        if not (tri_z[f] > near).all():
            continue
        (x0, y0), (x1, y1), (x2, y2) = tri_uv[f].tolist()
        w0 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        w1 = (x0 - x2) * (py - y2) - (y0 - y2) * (px - x2)
        w2 = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
        area = w0 + w1 + w2
        inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        inside = inside & (area.abs() > 0)
        iz = (w0 / tri_z[f, 0] + w1 / tri_z[f, 1] + w2 / tri_z[f, 2]) / area
        better = inside & (iz >= 1.0 / far) & ((best_id < 0) | (iz > best_iz))
        best_iz = torch.where(better, iz, best_iz)
        best_id = torch.where(better, torch.full_like(best_id, f), best_id)
    depth = torch.where(best_id >= 0, 1.0 / best_iz.clamp(min=1e-12), torch.zeros_like(best_iz))
    return best_id.view(H, W), depth.view(H, W)


def test_mesh_rasterizer_matches_brute_force():
    W, H = 83, 61
    verts, faces = _random_mesh()
    viewmats, Ks = _camera(W, H)
    out = rasterize_mesh(verts, faces, viewmats, Ks, W, H, near_plane=0.1, far_plane=100.0)
    ref_ids, ref_depth = _mesh_reference(verts, faces, viewmats, Ks, W, H, 0.1, 100.0)

    ids = out["face_ids"][0].long()
    same = ids == ref_ids
    # ties at shared edges / pixel centers exactly on an edge may differ
    assert same.float().mean().item() > 0.995, same.float().mean().item()
    covered = same & (ids >= 0)
    assert covered.sum().item() > 0.2 * W * H
    torch.testing.assert_close(out["depth"][0, ..., 0][covered], ref_depth[covered], rtol=1e-4, atol=1e-4)
    assert (out["depth"][0, ..., 0][ids < 0] == 0).all()
    assert (out["mask"][0, ..., 0] == (ids >= 0).float()).all()
    # perspective-correct barycentrics sum to one
    bary_sum = out["bary"][0].sum(-1)[ids >= 0]
    torch.testing.assert_close(bary_sum, torch.ones_like(bary_sum))


def test_mesh_rasterizer_gradients():
    """Autograd gradients of depth and interpolated attributes match finite differences."""
    W, H = 64, 48
    dtype = torch.float64
    # two overlapping triangles at different depths
    verts = torch.tensor([
        [-0.9, -0.6, 3.0], [0.8, -0.5, 3.4], [0.0, 0.7, 2.8],
        [-0.3, -0.2, 2.5], [0.6, 0.1, 2.6], [0.1, 0.5, 2.4],
    ], dtype=dtype, device=DEVICE)
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device=DEVICE)
    colors = torch.rand(6, 3, dtype=dtype, device=DEVICE)
    viewmats, Ks = _camera(W, H, dtype)

    base = rasterize_mesh(verts, faces, viewmats, Ks, W, H, attributes=colors)
    ids = base["face_ids"][0]
    # pixels whose 5x5 neighborhood is covered by the same face (stable under small perturbations)
    ids_f = ids.double()[None, None]
    stable = (F.max_pool2d(ids_f, 5, 1, 2) == F.max_pool2d(-ids_f, 5, 1, 2).neg())[0, 0] & (ids >= 0)
    assert stable.sum().item() > 30
    weights = torch.rand(1, H, W, 4, dtype=dtype, device=DEVICE) * stable[None, ..., None]

    def fn(v, c):
        out = rasterize_mesh(v, faces, viewmats, Ks, W, H, attributes=c)
        return (torch.cat([out["depth"], out["attributes"]], dim=-1) * weights).sum()

    v = verts.clone().requires_grad_(True)
    c = colors.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(fn, (v, c), eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("width,height", [(320, 160), (160, 320)])
def test_mesh_is_pixel_aligned_with_gaussians(width, height):
    """
    Facet centers lie on the mesh surface, so when projected with the camera (as
    gsplat does) they must land on mesh pixels, and they can never be in front
    of the mesh z-buffer.
    """
    from gmesh.scenes.mesh import Mesh
    from gmesh.scenes.cameras import PerspectiveCamera
    from gmesh.utils.mesh2gaussians_utils import mesh_to_gaussians

    vertices, faces = load_spot()
    mesh = Mesh(verts=vertices, faces=faces)
    gaussians = mesh_to_gaussians(mesh=mesh, method="facet")

    camera = PerspectiveCamera(image_width=width, image_height=height, fov_x=40,
                               fov_y=40 * height / width, z_near=0.1, z_far=100, device=DEVICE)
    camera_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 7.0]], device=DEVICE)

    _, depth_m, alpha_m = MeshRenderer(camera=camera).render(mesh=mesh, camera_pose=camera_pose)

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


def test_matches_pytorch3d_renderer():
    """Same coverage and depth as the former PyTorch3D-based renderer (if installed)."""
    pytest.importorskip("pytorch3d")
    from gmesh.scenes.mesh import Mesh
    from gmesh.scenes.cameras import PerspectiveCamera
    from gmesh.renderer.renderer_pytorch3d import CustomMeshRenderer

    vertices, faces = load_spot()
    mesh = Mesh(verts=vertices, faces=faces)
    camera = PerspectiveCamera(image_width=320, image_height=160, fov_x=40, fov_y=20,
                               z_near=0.1, z_far=100, device=DEVICE)
    camera_pose = torch.tensor([[0.3, 0.2, 0.0, 0.0, 0.0, 7.0]], device=DEVICE)

    _, depth_a, alpha_a = MeshRenderer(camera=camera).render(mesh=mesh, camera_pose=camera_pose)
    _, depth_b, alpha_b = CustomMeshRenderer(camera=camera).render(mesh=mesh, camera_pose=camera_pose)
    agree = (alpha_a == alpha_b).float().mean().item()
    assert agree > 0.99, agree
    both = (alpha_a > 0) & (alpha_b > 0)
    close = ((depth_a - depth_b).abs() <= 0.01 * depth_b)[both].float().mean().item()
    assert close > 0.98, close


def _spot_scene(W=256, H=128):
    from gmesh.scenes.mesh import Mesh
    from gmesh.scenes.cameras import PerspectiveCamera
    from gmesh.renderer.shader import simple_phone_shading

    vertices, faces = load_spot()
    mesh = Mesh(verts=vertices, faces=faces)
    mesh.colors = simple_phone_shading(vertices=vertices, faces=faces)
    camera = PerspectiveCamera(image_width=W, image_height=H, fov_x=40, fov_y=40 * H / W,
                               z_near=0.1, z_far=100, device=DEVICE)
    camera_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 7.0]], device=DEVICE)
    return vertices, faces, mesh, camera, camera_pose


def _gaussians_on(vertices, faces, dz):
    from gmesh.scenes.mesh import Mesh
    from gmesh.utils.mesh2gaussians_utils import mesh_to_gaussians

    gaussians = mesh_to_gaussians(mesh=Mesh(verts=vertices.clone(), faces=faces), method="facet3", trainable=True)
    with torch.no_grad():
        gaussians.means[:, 2] += dz  # the camera looks along -z from z = 7
    return gaussians


@pytest.mark.parametrize("mode", ["hybrid", "zbuffer"])
def test_gaussians_behind_the_mesh_are_hidden(mode):
    from gmesh.renderer.gmesh_renderer import GmeshRenderer

    vertices, faces, mesh, camera, camera_pose = _spot_scene()
    renderer = GmeshRenderer(camera=camera, tile_size=16, bg_color=(1.0, 1.0, 1.0))
    gaussians = _gaussians_on(vertices, faces, dz=-3.0)

    ret = renderer.render(mesh=mesh, gaussians=gaussians, camera_pose=camera_pose, return_all=True, mode=mode)
    rgb_m, _, alpha_m = ret["mesh"]
    rgb, depth, alpha = ret["blend"]
    covered = alpha_m[..., 0] > 0
    torch.testing.assert_close(rgb[covered], rgb_m[covered])
    # but visible next to the mesh
    assert (rgb[~covered] - 1.0).abs().max().item() > 0.1
    assert torch.isfinite(rgb).all() and torch.isfinite(depth).all() and torch.isfinite(alpha).all()


def test_hybrid_equals_zbuffer_when_gaussians_are_well_in_front():
    from gmesh.renderer.gmesh_renderer import GmeshRenderer

    vertices, faces, mesh, camera, camera_pose = _spot_scene()
    renderer = GmeshRenderer(camera=camera, tile_size=16, bg_color=(1.0, 1.0, 1.0))
    gaussians = _gaussians_on(vertices, faces, dz=3.0)
    with torch.no_grad():
        rgb_h, depth_h, alpha_h = renderer.render(mesh=mesh, gaussians=gaussians, camera_pose=camera_pose, mode="hybrid")
        rgb_z, depth_z, alpha_z = renderer.render(mesh=mesh, gaussians=gaussians, camera_pose=camera_pose, mode="zbuffer")
    torch.testing.assert_close(rgb_h, rgb_z, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(alpha_h, alpha_z, rtol=1e-4, atol=1e-4)


def test_hybrid_gradients_reach_gaussians_and_mesh():
    from gmesh.renderer.gmesh_renderer import GmeshRenderer

    vertices, faces, mesh, camera, camera_pose = _spot_scene()
    renderer = GmeshRenderer(camera=camera, tile_size=16, bg_color=(1.0, 1.0, 1.0))
    # Gaussians lying on the surface: the soft split is active everywhere
    gaussians = _gaussians_on(vertices, faces, dz=0.0)
    mesh.verts = vertices.clone().requires_grad_(True)
    mesh.colors = mesh.colors.detach().clone().requires_grad_(True)

    rgb, depth, alpha = renderer.render(mesh=mesh, gaussians=gaussians, camera_pose=camera_pose)
    (rgb.mean() + 0.1 * depth.mean()).backward()
    for name in ("means", "quats", "scales", "opacities", "colors"):
        grad = getattr(gaussians, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum().item() > 0, name
    for name, grad in (("mesh verts", mesh.verts.grad), ("mesh colors", mesh.colors.grad)):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum().item() > 0, name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
