#
# Peizhi Yan
# Copyright (C) 2025
#

"""
Tests for the hybrid (soft, per-ray depth test) Gaussian/mesh compositing kernel.

Requires a CUDA GPU (and gsplat v1.5.2 for the comparison with gsplat's rasterizer).
The first run JIT-compiles the CUDA extension, which takes a minute.

Run with:
    python -m pytest tests -v
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from gmesh.renderer.rasterizer_gaussians import rasterize_to_pixels_hybrid

from common import (
    DEVICE, assert_close, leaves, pixel_centers, random_scene, random_weights, tile_membership,
)


INPUT_KEYS = ("means2d", "conics", "colors", "opacities", "planes")
THRESHOLD = 1.0 / 255.0


def _reference(means2d, conics, colors, opacities, planes, depth_limits, coverages,
               depths, member, W, H, softness, skip_covered_back):
    """
    Brute-force PyTorch version of the hybrid kernel (autograd for backward).
    Mirrors the CUDA kernel: front-to-back order, alpha threshold, 0.999 clamp,
    T <= 1e-4 termination of each layer, and the per-pixel split.
    """
    C, N = means2d.shape[:2]
    px, py = pixel_centers(W, H)

    outs = {"front": [], "T_front": [], "back": [], "T_back": []}
    for c in range(C):
        D = depth_limits[c].flatten()
        has_mesh = torch.isfinite(D)
        D_safe = torch.where(has_mesh, D, torch.zeros_like(D))
        cov = coverages[c].flatten()
        back_active = has_mesh & ~(skip_covered_back & (cov >= 1.0))

        T_f = torch.ones_like(px)
        T_b = torch.ones_like(px)
        acc_f = torch.zeros(px.shape[0], colors.shape[-1], device=DEVICE)
        acc_b = torch.zeros(px.shape[0], colors.shape[-1], device=DEVICE)
        done = torch.zeros_like(px, dtype=torch.bool)
        zero = torch.zeros_like(px)

        for n in torch.argsort(depths[c]).tolist():
            dx = means2d[c, n, 0] - px
            dy = means2d[c, n, 1] - py
            a2, b2, c2 = conics[c, n, 0], conics[c, n, 1], conics[c, n, 2]
            sigma = 0.5 * (a2 * dx * dx + c2 * dy * dy) + b2 * dx * dy
            alpha = torch.clamp_max(opacities[c, n] * torch.exp(-sigma), 0.999)
            valid = member[c, :, n] & (sigma >= 0) & (alpha >= THRESHOLD) & ~done

            # split at the mesh depth, with the Gaussian's depth model at this pixel
            z_c, g_x, g_y, sigma_z = planes[c, n].unbind()
            z_star = z_c - (g_x * dx + g_y * dy)
            s = (D_safe - z_star) / (sigma_z * softness)
            frac = torch.where(has_mesh, torch.special.ndtr(s), torch.ones_like(s))

            alpha_f = alpha * frac
            use_f = valid & (alpha_f >= THRESHOLD)
            next_Tf = T_f * (1.0 - alpha_f)
            term_f = use_f & (next_Tf <= 1e-4)
            done = done | term_f
            contrib_f = use_f & ~term_f
            acc_f = acc_f + torch.where(contrib_f, alpha_f * T_f, zero)[:, None] * colors[c, n][None]
            T_f = torch.where(contrib_f, next_Tf, T_f)

            alpha_b = alpha * (1.0 - frac)
            use_b = valid & ~term_f & back_active & (alpha_b >= THRESHOLD)
            next_Tb = T_b * (1.0 - alpha_b)
            term_b = use_b & (next_Tb <= 1e-4)
            back_active = back_active & ~term_b
            contrib_b = use_b & ~term_b
            acc_b = acc_b + torch.where(contrib_b, alpha_b * T_b, zero)[:, None] * colors[c, n][None]
            T_b = torch.where(contrib_b, next_Tb, T_b)

        outs["front"].append(acc_f.view(H, W, -1))
        outs["T_front"].append(T_f.view(H, W, 1))
        outs["back"].append(acc_b.view(H, W, -1))
        outs["T_back"].append(T_b.view(H, W, 1))
    return {k: torch.stack(v) for k, v in outs.items()}


def _run(scene, depth_limits, coverages, softness=1.0, inputs=None, skip_covered_back=True):
    if inputs is None:
        inputs = leaves(scene, INPUT_KEYS)
    out = rasterize_to_pixels_hybrid(
        *inputs, depth_limits, coverages, scene["W"], scene["H"], scene["tile_size"],
        scene["isect_offsets"], scene["flatten_ids"], softness=softness, skip_covered_back=skip_covered_back,
    )
    return out, inputs


def _loss(out, scene, channels):
    w_f, w_tf = random_weights(scene, channels, seed=1)
    w_b, w_tb = random_weights(scene, channels, seed=2)
    return ((out["front"] * w_f).sum() + (out["T_front"] * w_tf).sum()
            + (out["back"] * w_b).sum() + (out["T_back"] * w_tb).sum())


@pytest.mark.parametrize("channels", [3, 4])
def test_no_mesh_matches_gsplat(channels):
    """Without a mesh, the front layer is plain 3DGS (forward and gradients), as gsplat's rasterizer."""
    gsplat_wrapper = pytest.importorskip("gsplat.cuda._wrapper")
    rasterize_to_pixels = gsplat_wrapper.rasterize_to_pixels
    scene = random_scene(channels=channels)
    W, H, C, ts = scene["W"], scene["H"], scene["C"], scene["tile_size"]
    limits = torch.full((C, H, W), float("inf"), device=DEVICE)
    coverages = torch.zeros(C, H, W, device=DEVICE)
    w_c, w_a = random_weights(scene, channels)

    ref_in = leaves(scene)
    ref_c, ref_a = rasterize_to_pixels(*ref_in, W, H, ts, scene["isect_offsets"], scene["flatten_ids"], packed=False)
    ((ref_c * w_c).sum() + (ref_a * w_a).sum()).backward()

    out, our_in = _run(scene, limits, coverages)
    ((out["front"] * w_c).sum() + ((1.0 - out["T_front"]) * w_a).sum()).backward()

    assert_close(out["front"], ref_c, "front", atol_scale=1e-6)
    assert_close(1.0 - out["T_front"], ref_a, "alpha", atol_scale=1e-6)
    assert out["back"].abs().max().item() == 0.0
    assert (out["T_back"] == 1.0).all()
    for name, ours, ref in zip(("means2d", "conics", "colors", "opacities"), our_in[:4], ref_in):
        assert_close(ours.grad, ref.grad, f"grad {name}", atol_scale=1e-5)
    # the depth model gets no gradient without a mesh
    assert our_in[4].grad.abs().max().item() == 0.0


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("softness", [1.0, 0.3])
@pytest.mark.parametrize("skip_covered_back", [True, False])
def test_matches_reference(channels, softness, skip_covered_back):
    """With a mesh, compare all outputs and gradients to a brute-force reference."""
    scene = random_scene(channels=channels, seed=3)
    W, H, C = scene["W"], scene["H"], scene["C"]

    # a slanted surface through the Gaussians, no mesh in some pixels, and
    # partial coverage in others
    xs = torch.arange(W, device=DEVICE).float()
    limits = (2.6 + 0.8 * xs / W).view(1, 1, W).repeat(C, H, 1)
    limits[:, H // 2:, : W // 3] = float("inf")
    coverages = torch.isfinite(limits).float()
    coverages[:, : H // 3, W // 2:] = 0.5
    limits = limits.clone().requires_grad_(True)

    member = tile_membership(scene)
    ref_in = leaves(scene, INPUT_KEYS)
    ref_limits = limits.detach().clone().requires_grad_(True)
    ref = _reference(*ref_in, ref_limits, coverages.detach(), scene["depths"], member, W, H,
                     softness, skip_covered_back=skip_covered_back)
    _loss(ref, scene, channels).backward()

    out, our_in = _run(scene, limits, coverages, softness=softness, skip_covered_back=skip_covered_back)
    _loss(out, scene, channels).backward()

    # the mesh actually splits some Gaussians
    assert out["back"].abs().sum().item() > 0
    for key in ("front", "T_front", "back", "T_back"):
        assert_close(out[key], ref[key], key)
    for name, ours, theirs in zip(INPUT_KEYS, our_in, ref_in):
        assert_close(ours.grad, theirs.grad, f"grad {name}")
    finite = torch.isfinite(limits.detach())
    assert_close(limits.grad[finite], ref_limits.grad[finite], "grad depth_limits")
    assert limits.grad[~finite].abs().max().item() == 0.0


def test_split_is_continuous():
    """
    Moving the mesh surface across the Gaussians changes the image continuously
    (no hard cut): with 4x smaller depth steps, the largest change per step
    shrinks accordingly (a hard cut would keep it constant).
    """
    scene = random_scene(seed=4)
    W, H, C = scene["W"], scene["H"], scene["C"]
    coverages = torch.ones(C, H, W, device=DEVICE)
    inputs = [scene[k] for k in INPUT_KEYS]

    def max_step(n_steps):
        prev, worst = None, 0.0
        for d in torch.linspace(2.4, 3.6, n_steps).tolist():
            limits = torch.full((C, H, W), d, device=DEVICE)
            out, _ = _run(scene, limits, coverages, inputs=inputs)
            if prev is not None:
                worst = max(worst, (out["front"] - prev).abs().max().item())
            prev = out["front"]
        return worst

    coarse, fine = max_step(121), max_step(481)
    assert coarse > 0
    assert fine < 0.5 * coarse, (coarse, fine)


def test_occlusion_culling_is_exact():
    """Culling hidden Gaussians in fully covered tiles does not change the result."""
    from gmesh.renderer.rasterizer_gaussians import rasterization_hybrid

    g = torch.Generator().manual_seed(5)
    N, W, H = 300, 96, 64
    means = ((torch.rand(N, 3, generator=g) * 2 - 1) * torch.tensor([1.5, 1.0, 2.0]) + torch.tensor([0.0, 0.0, 4.0])).to(DEVICE)
    quats = torch.randn(N, 4, generator=g).to(DEVICE)
    scales = (torch.rand(N, 3, generator=g) * 0.08 + 0.01).to(DEVICE)
    opacities = (torch.rand(N, generator=g) * 0.8 + 0.1).to(DEVICE)
    shs = torch.rand(N, 1, 3, generator=g).to(DEVICE)
    viewmats = torch.eye(4, device=DEVICE)[None]
    Ks = torch.tensor([[70.0, 0.0, W / 2], [0.0, 70.0, H / 2], [0.0, 0.0, 1.0]], device=DEVICE)[None]
    # a wall at depth 3 covering the left 2/3 of the image, slanted, with a partially covered column
    xs = torch.arange(W, device=DEVICE).float()
    limits = (2.8 + 0.2 * xs / W).view(1, 1, W).repeat(1, H, 1)
    coverages = torch.ones(1, H, W, device=DEVICE)
    limits[:, :, 2 * W // 3:] = float("inf")
    coverages[:, :, 2 * W // 3:] = 0.0
    coverages[:, :, 2 * W // 3 - 1] = 0.5

    outs = {}
    for cull in (False, True):
        params = [t.clone().requires_grad_(True) for t in (means, quats, scales, opacities, shs)]
        layers = rasterization_hybrid(*params[:4], params[4], 0, viewmats, Ks, W, H, limits, coverages,
                                      cull_occluded=cull)
        loss = sum((layers[k] * torch.randn(layers[k].shape, generator=torch.Generator().manual_seed(7)).to(DEVICE)).sum()
                   for k in ("front", "T_front", "back", "T_back"))
        loss.backward()
        outs[cull] = (layers, [p.grad for p in params])

    assert outs[True][0]["n_isects_rendered"] < outs[False][0]["n_isects_rendered"]
    for k in ("front", "T_front", "back", "T_back"):
        torch.testing.assert_close(outs[True][0][k], outs[False][0][k])
    for name, ours, ref in zip(("means", "quats", "scales", "opacities", "shs"), outs[True][1], outs[False][1]):
        assert_close(ours, ref, f"grad {name}", rtol=1e-4, atol_scale=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
