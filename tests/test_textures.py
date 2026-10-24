#
# Peizhi Yan
# Copyright (C) 2026
#

"""
Tests for textured meshes: texture sampling in the mesh rasterizer (bilinear and
mipmap filters), its gradients, and textured meshes in MeshRenderer / GmeshRenderer.

Requires a CUDA GPU.

Run with:
    python -m pytest tests -v
"""

import math
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from gmesh.renderer.rasterizer_mesh import rasterize_mesh
from gmesh.utils.mesh_io import load_obj, load_texture

from common import DEVICE, ROOT

SPOT_OBJ = os.path.join(ROOT, "assets", "data", "spot", "spot_triangulated.obj")
SPOT_TEXTURE = os.path.join(ROOT, "assets", "data", "spot", "spot_texture.png")


def _camera(W, H, dtype=torch.float32):
    viewmats = torch.eye(4, dtype=dtype, device=DEVICE)[None]
    Ks = torch.tensor([[50.0, 0.0, W / 2], [0.0, 50.0, H / 2], [0.0, 0.0, 1.0]], dtype=dtype, device=DEVICE)[None]
    return viewmats, Ks


def _ramp(uv):
    """a linear function of the texture coordinates, [..., 2] -> [..., 3]"""
    u, v = uv[..., 0], uv[..., 1]
    return torch.stack([u, v, 0.3 + 0.5 * u - 0.2 * v], dim=-1)


def _ramp_texture(Wt, Ht, dtype=torch.float32):
    """texture whose texel centers hold _ramp (row 0 at the top: v = 1 - (y + 0.5) / Ht)"""
    x = (torch.arange(Wt, dtype=dtype, device=DEVICE) + 0.5) / Wt
    y = 1.0 - (torch.arange(Ht, dtype=dtype, device=DEVICE) + 0.5) / Ht
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    return _ramp(torch.stack([gx, gy], dim=-1))  # [Ht, Wt, 3]


def _random_mesh(F_=30, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    centers = (torch.rand(F_, 1, 3, generator=g) * 2 - 1) * torch.tensor([0.8, 0.5, 0.6]) + torch.tensor([0.0, 0.0, 3.0])
    verts = (centers + torch.randn(F_, 3, 3, generator=g) * 0.3).reshape(-1, 3)
    faces = torch.arange(F_ * 3).view(F_, 3)
    # texture coordinates per face corner, indexed differently from the vertices
    uvs = torch.rand(F_ * 3, 2, generator=g) * 0.8 + 0.1
    faces_uv = torch.randperm(F_ * 3, generator=g).view(F_, 3)
    return [t.to(DEVICE) for t in (verts.to(dtype), faces, uvs.to(dtype), faces_uv)]


def _plane(size=0.5, z=3.0, tilt=0.15):
    """a square [-size, size]^2 at depth z (slightly tilted), with uv covering [0, 1]^2"""
    verts = torch.tensor([[-size, -size, z - tilt], [size, -size, z - tilt], [size, size, z + tilt], [-size, size, z + tilt]],
                         device=DEVICE)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=DEVICE)
    # image y points down, OBJ v points up
    uvs = torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], device=DEVICE)
    return verts, faces, uvs


def test_load_spot():
    verts, faces, uvs, faces_uv = load_obj(SPOT_OBJ)
    assert verts.shape == (2930, 3) and faces.shape == (5856, 3)
    assert uvs.shape == (3225, 2) and faces_uv.shape == (5856, 3)  # uv seams: more uvs than vertices
    assert faces.max().item() == 2929 and faces_uv.max().item() == 3224
    assert ((uvs > -0.1) & (uvs < 1.1)).all()  # (a few uvs are slightly outside [0, 1]: clamped to the border)
    texture = load_texture(SPOT_TEXTURE)
    assert texture.shape == (1024, 1024, 3) and 0.0 <= texture.min().item() and texture.max().item() <= 1.0


@pytest.mark.parametrize("antialias", [False, True])
def test_bilinear_texture_matches_vertex_colors(antialias):
    """A linear texture is reproduced exactly by bilinear sampling: same image as the vertex colors."""
    W, H = 83, 61
    verts, faces, uvs, faces_uv = _random_mesh()
    viewmats, Ks = _camera(W, H)
    # vertex colors equal to the texture function at the corners (each vertex is the corner of one face)
    colors = torch.zeros_like(verts)
    colors[faces.flatten()] = _ramp(uvs[faces_uv.flatten()])

    ref = rasterize_mesh(verts, faces, viewmats, Ks, W, H, attributes=colors, antialias=antialias)
    out = rasterize_mesh(verts, faces, viewmats, Ks, W, H, antialias=antialias, texture=_ramp_texture(32, 24),
                         texcoords=uvs, faces_uv=faces_uv, texture_filter="bilinear")
    assert (out["face_ids"] >= 0).sum().item() > 500
    torch.testing.assert_close(out["attributes"], ref["attributes"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out["color"], ref["color"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out["alpha"], ref["alpha"])
    # interpolated texture coordinates
    covered = out["face_ids"][0] >= 0
    torch.testing.assert_close(_ramp(out["texcoords"][0][covered]), ref["attributes"][0][covered], atol=1e-5, rtol=1e-5)


def _checker(n=64, period=2):
    i = torch.arange(n, device=DEVICE)
    board = (((i[:, None] // (period // 2)) + (i[None, :] // (period // 2))) % 2).float()
    return board[..., None].repeat(1, 1, 3)


def test_mipmap_antialiases_minified_texture():
    """A minified fine checkerboard: mipmapping is close to a supersampled reference, bilinear aliases."""
    W, H = 48, 40
    verts, faces, uvs = _plane()
    viewmats, Ks = _camera(W, H)
    texture = _checker(64)  # ~4 texels per pixel
    kw = dict(texture=texture, texcoords=uvs)
    mip = rasterize_mesh(verts, faces, viewmats, Ks, W, H, texture_filter="mipmap", **kw)
    bil = rasterize_mesh(verts, faces, viewmats, Ks, W, H, texture_filter="bilinear", **kw)
    ss = 8
    Ks_ss = Ks.clone()
    Ks_ss[:, :2, :] *= ss
    sup = rasterize_mesh(verts, faces, viewmats, Ks_ss, W * ss, H * ss, texture_filter="bilinear", **kw)
    ref = sup["attributes"].view(1, H, ss, W, ss, 3).mean(dim=(2, 4))
    inside = sup["mask"].view(1, H, ss, W, ss).amin(dim=(2, 4)) > 0  # pixels fully inside the plane
    assert inside.sum().item() > 100
    err_mip = (mip["attributes"] - ref)[inside].abs().mean().item()
    err_bil = (bil["attributes"] - ref)[inside].abs().mean().item()
    assert err_mip < 0.25 * err_bil, (err_mip, err_bil)
    assert err_mip < 0.05, err_mip


def test_mipmap_spreads_texture_gradients():
    """With a minified texture, mipmapping gives gradients to (almost) all texels, bilinear only to a few."""
    W, H = 48, 40
    verts, faces, uvs = _plane(tilt=0.0)
    viewmats, Ks = _camera(W, H)
    texels = {}
    for filt in ("mipmap", "bilinear"):
        texture = torch.full((64, 64, 3), 0.5, device=DEVICE, requires_grad=True)
        out = rasterize_mesh(verts, faces, viewmats, Ks, W, H, texture=texture, texcoords=uvs, texture_filter=filt)
        out["attributes"].sum().backward()
        texels[filt] = (texture.grad.abs().sum(dim=-1) > 0).float().mean().item()
    assert texels["mipmap"] > 0.95, texels
    assert texels["bilinear"] < 0.5, texels


@pytest.mark.parametrize("texture_filter", ["bilinear", "mipmap"])
def test_texture_gradients(texture_filter):
    """Autograd gradients w.r.t. the texture (and, for bilinear, the texture coordinates) match finite differences."""
    W, H = 40, 32
    dtype = torch.float64
    verts, faces, uvs, faces_uv = _random_mesh(F_=6, seed=2, dtype=dtype)
    viewmats, Ks = _camera(W, H, dtype)
    g = torch.Generator().manual_seed(3)
    texture = torch.rand(7, 9, 3, generator=g, dtype=dtype).to(DEVICE)
    weights = torch.rand(1, H, W, 3, generator=g, dtype=dtype).to(DEVICE)

    def fn(tex, tc):
        out = rasterize_mesh(verts, faces, viewmats, Ks, W, H, texture=tex, texcoords=tc, faces_uv=faces_uv,
                             texture_filter=texture_filter)
        return (out["attributes"] * weights).sum()

    tex = texture.clone().requires_grad_(True)
    tc = uvs.clone().requires_grad_(texture_filter == "bilinear")  # the mipmap level is not differentiated
    assert torch.autograd.gradcheck(fn, (tex, tc), eps=1e-6, atol=1e-6, rtol=1e-4)


def _spot_textured():
    from gmesh.scenes.mesh import Mesh
    from gmesh.scenes.cameras import PerspectiveCamera

    verts, faces, uvs, faces_uv = load_obj(SPOT_OBJ, device=DEVICE)
    theta = math.pi * 0.8
    R = torch.tensor([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]],
                     device=DEVICE)
    texture = load_texture(SPOT_TEXTURE, device=DEVICE)
    mesh = Mesh(verts=verts @ R.T, faces=faces, uvs=uvs, faces_uv=faces_uv, texture=texture)
    W, H = 256, 192
    camera = PerspectiveCamera(image_width=W, image_height=H, fov_x=40, fov_y=40 * H / W,
                               z_near=0.1, z_far=100, device=DEVICE)
    camera_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 3.5]], device=DEVICE)
    return mesh, camera, camera_pose


def test_textured_mesh_renderer():
    from gmesh.renderer.rasterizer_mesh import MeshRenderer

    mesh, camera, camera_pose = _spot_textured()
    renderer = MeshRenderer(camera=camera, bg_color=(1.0, 1.0, 1.0))
    mesh.texture = mesh.texture.clone().requires_grad_(True)
    mesh.verts = mesh.verts.clone().requires_grad_(True)
    rgb, depth, alpha = renderer.render(mesh, camera_pose)
    assert (alpha > 0).float().mean().item() > 0.1
    # the texture is used, not the (default gray) vertex colors
    covered = alpha[0, ..., 0] >= 1
    assert (rgb[0][covered] - 0.5).abs().amax(dim=-1).mean().item() > 0.1
    rgb.mean().backward()
    for name, t in (("texture", mesh.texture), ("verts", mesh.verts)):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum().item() > 0, name


def test_textured_mesh_in_hybrid_renderer():
    from gmesh.renderer.gmesh_renderer import GmeshRenderer
    from gmesh.utils.mesh2gaussians_utils import mesh_to_gaussians
    from gmesh.scenes.mesh import Mesh

    mesh, camera, camera_pose = _spot_textured()
    renderer = GmeshRenderer(camera=camera, bg_color=(1.0, 1.0, 1.0))
    # without Gaussians in the way, the hybrid render equals the mesh render
    gaussians = mesh_to_gaussians(Mesh(verts=mesh.verts.clone(), faces=mesh.faces), method="facet", trainable=True)
    with torch.no_grad():
        gaussians.means[:, 2] -= 3.0  # behind the mesh, hidden
        ret = renderer.render(mesh=mesh, gaussians=gaussians, camera_pose=camera_pose, return_all=True)
    rgb_m, _, alpha_m = ret["mesh"]
    rgb, _, _ = ret["blend"]
    covered = alpha_m[..., 0] >= 1
    torch.testing.assert_close(rgb[covered], rgb_m[covered])

    with torch.no_grad():
        gaussians.means[:, 2] += 3.0  # on the surface
    mesh.texture = mesh.texture.clone().requires_grad_(True)
    rgb, depth, alpha = renderer.render(mesh=mesh, gaussians=gaussians, camera_pose=camera_pose)
    rgb.mean().backward()
    assert mesh.texture.grad is not None and mesh.texture.grad.abs().sum().item() > 0
    assert gaussians.means.grad is not None and gaussians.means.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Lighting (per-pixel Phong shading of vertex colors or textures)
# ---------------------------------------------------------------------------

def _facing_plane(z=3.0, size=0.5):
    """a square at depth z facing the camera (normal towards the camera, which is at the origin looking along +z)"""
    verts = torch.tensor([[-size, -size, z], [size, -size, z], [size, size, z], [-size, size, z]], device=DEVICE)
    faces = torch.tensor([[0, 2, 1], [0, 3, 2]], device=DEVICE)
    uvs = torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], device=DEVICE)
    return verts, faces, uvs


@pytest.mark.parametrize("source", ["texture", "vertex_colors"])
def test_lighting_matches_phong(source):
    """Center pixel of a plane lit from the camera: ambient * c + diffuse * c + specular; lit from behind: ambient * c."""
    from gmesh.scenes.light import PointLight

    W, H = 49, 41  # the principal point is the center of pixel (24, 20)
    verts, faces, uvs = _facing_plane()
    viewmats, Ks = _camera(W, H)
    c = torch.tensor([0.6, 0.5, 0.4], device=DEVICE)
    if source == "texture":
        kw = dict(texture=c.view(1, 1, 3).repeat(8, 8, 1), texcoords=uvs)
    else:
        kw = dict(attributes=c.repeat(4, 1))
    ambient, diffuse, specular = (0.2, 0.25, 0.3), (0.5, 0.4, 0.3), (0.1, 0.05, 0.0)
    expected = {
        "front": torch.tensor(ambient, device=DEVICE) * c + torch.tensor(diffuse, device=DEVICE) * c
                 + torch.tensor(specular, device=DEVICE),
        "behind": torch.tensor(ambient, device=DEVICE) * c,
    }
    for where, position in (("front", (0.0, 0.0, 0.0)), ("behind", (0.0, 0.0, 6.0))):
        light = PointLight(position=position, ambient=ambient, diffuse=diffuse, specular=specular, shininess=10.0,
                           device=DEVICE)
        out = rasterize_mesh(verts, faces, viewmats, Ks, W, H, light=light, **kw)
        torch.testing.assert_close(out["color"][0, 20, 24], expected[where], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(out["attributes"][0, 20, 24], c, atol=1e-5, rtol=1e-5)  # unlit


def test_lighting_gradients():
    """Autograd gradients w.r.t. the light, the texture and the vertex positions match finite differences."""
    from gmesh.scenes.light import PointLight

    W, H = 40, 32
    dtype = torch.float64
    # a curved grid facing the camera (smooth, interpolated vertex normals)
    n = 3
    xs = torch.linspace(-1.0, 1.0, n + 1, dtype=dtype, device=DEVICE)
    gy, gx = torch.meshgrid(xs * 0.8, xs, indexing="ij")
    verts = torch.stack([gx, gy, 3.0 + 0.3 * gx * gx - 0.2 * gy], dim=-1).reshape(-1, 3)
    idx = torch.arange((n + 1) * (n + 1), device=DEVICE).view(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].flatten(), idx[:-1, 1:].flatten(), idx[1:, :-1].flatten(), idx[1:, 1:].flatten()
    faces = torch.cat([torch.stack([a, d, b], -1), torch.stack([a, c, d], -1)])
    uvs = torch.stack([(gx.flatten() + 1) / 2, 1 - (gy.flatten() / 0.8 + 1) / 2], dim=-1) * 0.8 + 0.1
    faces_uv = faces
    viewmats, Ks = _camera(W, H, dtype)
    g = torch.Generator().manual_seed(5)
    texture = (torch.rand(5, 6, 3, generator=g, dtype=dtype) * 0.5 + 0.2).to(DEVICE)

    base = rasterize_mesh(verts, faces, viewmats, Ks, W, H, texture=texture, texcoords=uvs, faces_uv=faces_uv)
    ids = base["face_ids"][0]
    ids_f = ids.double()[None, None]
    # pixels whose 3x3 neighborhood is covered by the same face (stable under the finite differences)
    stable = (torch.nn.functional.max_pool2d(ids_f, 3, 1, 1) == torch.nn.functional.max_pool2d(-ids_f, 3, 1, 1).neg())[0, 0] & (ids >= 0)
    assert stable.sum().item() > 100
    weights = torch.rand(1, H, W, 3, generator=g, dtype=dtype).to(DEVICE) * stable[None, ..., None]

    def fn(v, tex, position, ambient, diffuse, specular):
        light = PointLight(position=position, ambient=ambient, diffuse=diffuse, specular=specular, shininess=4.0,
                           device=DEVICE)
        out = rasterize_mesh(v, faces, viewmats, Ks, W, H, texture=tex, texcoords=uvs, faces_uv=faces_uv,
                             texture_filter="bilinear", light=light)
        return (out["color"] * weights).sum()

    inputs = [verts, texture,
              torch.tensor([0.3, -0.2, 0.5], dtype=dtype, device=DEVICE),
              torch.tensor([0.2, 0.15, 0.1], dtype=dtype, device=DEVICE),
              torch.tensor([0.4, 0.3, 0.35], dtype=dtype, device=DEVICE),
              torch.tensor([0.1, 0.2, 0.05], dtype=dtype, device=DEVICE)]
    inputs = [t.clone().requires_grad_(True) for t in inputs]
    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_lit_textured_mesh_is_trainable():
    """A loaded texture (trainable=True) and the light get gradients through the lit render."""
    from gmesh.renderer.rasterizer_mesh import MeshRenderer
    from gmesh.scenes.light import PointLight
    from gmesh.utils.mesh_io import load_textured_mesh

    _, camera, camera_pose = _spot_textured()
    mesh = load_textured_mesh(SPOT_OBJ, SPOT_TEXTURE, device=DEVICE, trainable=True)
    assert isinstance(mesh.texture, torch.nn.Parameter)
    light = PointLight(position=(1.0, 1.0, 5.0), specular=(0.2, 0.2, 0.2), device=DEVICE)
    for t in (light.position, light.ambient, light.diffuse, light.specular):
        t.requires_grad_(True)

    renderer = MeshRenderer(camera=camera, bg_color=(1.0, 1.0, 1.0))
    rgb_lit, _, alpha = renderer.render(mesh, camera_pose, light=light)
    rgb_unlit, _, _ = renderer.render(mesh, camera_pose)
    covered = alpha[0, ..., 0] >= 1
    assert covered.sum().item() > 1000
    assert (rgb_lit[0][covered] - rgb_unlit[0][covered]).abs().mean().item() > 0.05  # the light changes the image
    rgb_lit.mean().backward()
    for name, t in (("texture", mesh.texture), ("position", light.position), ("ambient", light.ambient),
                    ("diffuse", light.diffuse), ("specular", light.specular)):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum().item() > 0, name


def test_lit_mesh_in_hybrid_renderer():
    from gmesh.renderer.gmesh_renderer import GmeshRenderer
    from gmesh.renderer.rasterizer_mesh import MeshRenderer
    from gmesh.scenes.light import PointLight
    from gmesh.utils.mesh2gaussians_utils import mesh_to_gaussians
    from gmesh.scenes.mesh import Mesh

    mesh, camera, camera_pose = _spot_textured()
    light = PointLight(position=(1.0, 1.0, 5.0), device=DEVICE)
    gaussians = mesh_to_gaussians(Mesh(verts=mesh.verts.clone(), faces=mesh.faces), method="facet")
    gaussians.means[:, 2] -= 3.0  # behind the mesh, hidden
    with torch.no_grad():
        rgb, _, _ = GmeshRenderer(camera=camera).render(mesh=mesh, gaussians=gaussians, camera_pose=camera_pose,
                                                        light=light)
        rgb_m, _, alpha_m = MeshRenderer(camera=camera).render(mesh, camera_pose, light=light)
    covered = alpha_m[..., 0] >= 1
    torch.testing.assert_close(rgb[covered], rgb_m[covered])


def test_renderer_default_light():
    """A light given to the renderer is used by default, and is the same as giving it to render()."""
    from gmesh.renderer.rasterizer_mesh import MeshRenderer
    from gmesh.scenes.light import PointLight

    mesh, camera, camera_pose = _spot_textured()
    light = PointLight(position=(1.0, 1.0, 5.0), specular=(0.2, 0.2, 0.2), device=DEVICE)
    with torch.no_grad():
        per_call, _, _ = MeshRenderer(camera=camera).render(mesh, camera_pose, light=light)
        default, _, _ = MeshRenderer(camera=camera, light=light).render(mesh, camera_pose)
        unlit, _, _ = MeshRenderer(camera=camera).render(mesh, camera_pose)
    torch.testing.assert_close(default, per_call)
    assert (default - unlit).abs().amax().item() > 0.1
    trainable = PointLight(device=DEVICE, trainable=True)
    assert all(isinstance(t, torch.nn.Parameter) for t in trainable.parameters())
