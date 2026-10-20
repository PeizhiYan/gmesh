#
# Peizhi Yan
# Copyright (C) 2026
#

"""
The projection, tile intersection and spherical harmonics kernels ported from
gsplat v1.5.2 (gmesh/renderer/csrc/gsplat, gmesh/renderer/gsplat_ops.py) must
give the same results as gsplat itself, forward and backward, and so must the
plain 3DGS rendering of GaussianRasterizer.

Requires a CUDA GPU and gsplat v1.5.2 (only for these comparisons; Gmesh itself
does not need gsplat):
    pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.5.2

Run with:
    python -m pytest tests -v
"""

import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)
gs = pytest.importorskip("gsplat.cuda._wrapper", reason="gsplat is needed to compare against it")
gs_rendering = pytest.importorskip("gsplat.rendering")

from gmesh.renderer import gsplat_ops

from common import DEVICE, assert_close


def _gaussians(N=500, seed=0, sh_degree=3):
    g = torch.Generator().manual_seed(seed)
    means = ((torch.rand(N, 3, generator=g) * 2 - 1) * torch.tensor([2.0, 1.5, 1.5]) + torch.tensor([0.0, 0.0, 4.0]))
    quats = torch.randn(N, 4, generator=g)  # not normalized on purpose
    scales = torch.exp(torch.empty(N, 3).uniform_(math.log(0.005), math.log(0.3), generator=g))
    opacities = torch.rand(N, generator=g) * 0.98 + 0.01
    shs = torch.randn(N, (sh_degree + 1) ** 2, 3, generator=g) * 0.3
    return [t.to(DEVICE) for t in (means, quats, scales, opacities, shs)]


def _cameras(C=2, W=96, H=64):
    viewmats = torch.eye(4).repeat(C, 1, 1)
    # second camera: rotated and shifted
    a = 0.3
    viewmats[1, :3, :3] = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    viewmats[1, :3, 3] = torch.tensor([0.4, -0.2, 0.3])
    Ks = torch.tensor([[80.0, 0.0, W / 2 + 1.5], [0.0, 75.0, H / 2 - 2.0], [0.0, 0.0, 1.0]]).repeat(C, 1, 1)
    return viewmats.to(DEVICE), Ks.to(DEVICE), W, H


def _leaves(*tensors):
    return [t.clone().requires_grad_(True) for t in tensors]


def test_projection_matches_gsplat():
    means, quats, scales, opacities, _ = _gaussians()
    viewmats, Ks, W, H = _cameras()
    g = torch.Generator().manual_seed(1)

    outs, grads = {}, {}
    for name in ("gsplat", "gmesh"):
        m, q, s, v = _leaves(means, quats, scales, viewmats)
        if name == "gsplat":
            radii, means2d, depths, conics, _ = gs.fully_fused_projection(
                m, None, q, s, v, Ks, W, H, eps2d=0.3, near_plane=0.01, far_plane=1e10,
                packed=False, calc_compensations=False, opacities=opacities,
            )
        else:
            radii, means2d, depths, conics = gsplat_ops.fully_fused_projection(
                m, q, s, opacities, v, Ks, W, H, eps2d=0.3, near_plane=0.01, far_plane=1e10,
            )
        valid = (radii > 0).all(dim=-1)
        w2 = torch.randn(means2d.shape, generator=g.manual_seed(2)).to(DEVICE)
        wd = torch.randn(depths.shape, generator=g.manual_seed(3)).to(DEVICE)
        wc = torch.randn(conics.shape, generator=g.manual_seed(4)).to(DEVICE)
        loss = ((means2d * w2)[valid].sum() + (depths * wd)[valid].sum() + (conics * wc)[valid].sum())
        loss.backward()
        outs[name] = (radii, means2d, depths, conics, valid)
        grads[name] = [t.grad for t in (m, q, s, v)]

    r_gs, m_gs, d_gs, c_gs, valid_gs = outs["gsplat"]
    r_gm, m_gm, d_gm, c_gm, valid_gm = outs["gmesh"]
    assert valid_gs.any() and (~valid_gs).any()  # some Gaussians are culled
    assert torch.equal(r_gm, r_gs)
    v = valid_gs
    torch.testing.assert_close(m_gm[v], m_gs[v], rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(d_gm[v], d_gs[v], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(c_gm[v], c_gs[v], rtol=1e-6, atol=1e-6)
    for name, ours, ref in zip(("means", "quats", "scales", "viewmats"), grads["gmesh"], grads["gsplat"]):
        assert_close(ours, ref, f"grad {name}", rtol=1e-5, atol_scale=1e-6)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_spherical_harmonics_match_gsplat(degree):
    _, _, _, _, shs = _gaussians(sh_degree=3)
    N = shs.shape[0]
    g = torch.Generator().manual_seed(5)
    dirs = torch.randn(N, 3, generator=g).to(DEVICE)
    masks = (torch.rand(N, generator=g) > 0.2).to(DEVICE)
    w = torch.randn(N, 3, generator=g).to(DEVICE)

    res = {}
    for name, fn in (("gsplat", gs.spherical_harmonics), ("gmesh", gsplat_ops.spherical_harmonics)):
        d, c = _leaves(dirs, shs)
        colors = fn(degree, d, c, masks=masks)
        (colors * w).sum().backward()
        res[name] = (colors, d.grad, c.grad)
    for i, what in enumerate(("colors", "grad dirs", "grad coeffs")):
        torch.testing.assert_close(res["gmesh"][i], res["gsplat"][i], rtol=1e-6, atol=1e-6, msg=lambda m: f"{what}: {m}")


def test_tile_intersection_matches_gsplat():
    means, quats, scales, opacities, _ = _gaussians(N=2000)
    viewmats, Ks, W, H = _cameras()
    C = viewmats.shape[0]
    with torch.no_grad():
        radii, means2d, depths, conics = gsplat_ops.fully_fused_projection(
            means, quats, scales, opacities, viewmats, Ks, W, H)
    tile_size = 16
    tw, th = math.ceil(W / tile_size), math.ceil(H / tile_size)

    t_gs, ids_gs, flat_gs = gs.isect_tiles(means2d, radii, depths, tile_size, tw, th, packed=False, n_cameras=C)
    t_gm, ids_gm, flat_gm = gsplat_ops.isect_tiles(means2d, radii, depths, tile_size, tw, th)
    assert torch.equal(t_gm, t_gs)
    assert torch.equal(ids_gm, ids_gs)
    assert torch.equal(flat_gm, flat_gs)
    assert torch.equal(gsplat_ops.isect_offset_encode(ids_gm, C, tw, th), gs.isect_offset_encode(ids_gs, C, tw, th))


@pytest.mark.parametrize("sh_degree", [0, 3])
def test_gaussian_rasterizer_matches_gsplat(sh_degree):
    """GaussianRasterizer (plain 3DGS through Gmesh's kernels) vs gsplat.rendering.rasterization."""
    from gmesh.scenes.gaussians import Gaussians
    from gmesh.scenes.cameras import PerspectiveCamera
    from gmesh.renderer.rasterizer_gaussians import GaussianRasterizer

    means, quats, scales, opacities, shs = _gaussians(N=800, sh_degree=sh_degree)
    W, H = 96, 64
    camera = PerspectiveCamera(image_width=W, image_height=H, fov_x=50, fov_y=50 * H / W,
                               z_near=0.1, z_far=100, device=DEVICE)
    camera_pose = torch.tensor([[0.1, -0.05, 0.0, 0.2, 0.1, 0.0]], device=DEVICE)
    bg = (0.2, 0.4, 0.6)
    g = torch.Generator().manual_seed(6)
    w_rgb = torch.randn(1, H, W, 3, generator=g).to(DEVICE)
    w_d = torch.randn(1, H, W, 1, generator=g).to(DEVICE)
    w_a = torch.randn(1, H, W, 1, generator=g).to(DEVICE)

    res = {}
    for name in ("gsplat", "gmesh"):
        m, q, s, o, c = _leaves(means, quats, scales, opacities, shs)
        if name == "gsplat":
            Ks = camera.get_intrinsic_matrix(batch_size=1)
            viewmats = camera.get_view_matrix(camera_pose)
            render, alpha, _ = gs_rendering.rasterization(
                m, q, s, o, c, viewmats, Ks, W, H, sh_degree=sh_degree, near_plane=0.1, far_plane=100,
                tile_size=16, backgrounds=torch.tensor([bg], device=DEVICE), render_mode="RGB+ED",
                packed=False,
            )
            rgb, depth = render[..., :3], render[..., 3:4]
        else:
            rasterizer = GaussianRasterizer(camera=camera, tile_size=16, bg_color=bg)
            rgb, depth, alpha = rasterizer.rasterize(Gaussians(m, q, s, o, c), camera_pose)
        covered = alpha > 1e-3
        ((rgb * w_rgb).sum() + (depth * w_d)[covered].sum() + (alpha * w_a).sum()).backward()
        res[name] = (rgb, depth, alpha, [t.grad for t in (m, q, s, o, c)])

    assert (res["gsplat"][2] > 0.5).float().mean().item() > 0.2  # the Gaussians cover the image
    assert_close(res["gmesh"][0], res["gsplat"][0], "rgb", atol_scale=1e-5)
    assert_close(res["gmesh"][2], res["gsplat"][2], "alpha", atol_scale=1e-5)
    covered = res["gsplat"][2] > 1e-3
    assert_close(res["gmesh"][1][covered], res["gsplat"][1][covered], "depth", atol_scale=1e-5)
    for name, ours, ref in zip(("means", "quats", "scales", "opacities", "sh"), res["gmesh"][3], res["gsplat"][3]):
        assert_close(ours, ref, f"grad {name}", rtol=1e-4, atol_scale=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
