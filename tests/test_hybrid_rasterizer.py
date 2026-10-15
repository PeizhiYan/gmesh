#
# Peizhi Yan
# Copyright (C) 2025
#

"""
Tests for the hybrid (soft, per-ray depth test) Gaussian/mesh compositing kernel.

Requires a CUDA GPU and gsplat v1.5.2.
The first run JIT-compiles the CUDA extension, which takes a minute.

Run with:
    python -m pytest tests -v
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from gsplat.cuda._wrapper import rasterize_to_pixels

from gmesh.renderer.rasterizer_hybrid import rasterize_to_pixels_hybrid

from common import (
    DEVICE, assert_close, leaves, pixel_centers, random_scene, random_weights, tile_membership,
)


INPUT_KEYS = ("means2d", "conics", "colors", "opacities", "mus", "precs")
THRESHOLD = 1.0 / 255.0


def _reference(means2d, conics, colors, opacities, mus, precs, depth_limits, coverages, Ks,
               depths, member, W, H, softness, skip_covered_back):
    """
    Brute-force PyTorch version of the hybrid kernel (autograd for backward).
    Mirrors the CUDA kernel: front-to-back order, alpha threshold, 0.999 clamp,
    T <= 1e-4 termination of each layer, and the per-ray split.
    """
    C, N = means2d.shape[:2]
    px, py = pixel_centers(W, H)

    outs = {"front": [], "T_front": [], "back": [], "T_back": []}
    for c in range(C):
        K = Ks[c]
        rx = (px - K[0, 2]) / K[0, 0]
        ry = (py - K[1, 2]) / K[1, 1]
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

            # split along the ray
            P = precs[c, n]
            mu = mus[c, n]
            Pd0 = P[0] * rx + P[1] * ry + P[2]
            Pd1 = P[1] * rx + P[3] * ry + P[4]
            Pd2 = P[2] * rx + P[4] * ry + P[5]
            a = rx * Pd0 + ry * Pd1 + Pd2
            b = mu[0] * Pd0 + mu[1] * Pd1 + mu[2] * Pd2
            s = (D_safe * a.sqrt() - b / a.sqrt()) / softness
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


def _run(scene, depth_limits, coverages, softness=1.0, inputs=None):
    if inputs is None:
        inputs = leaves(scene, INPUT_KEYS)
    out = rasterize_to_pixels_hybrid(
        *inputs, depth_limits, coverages, scene["Ks"], scene["W"], scene["H"], scene["tile_size"],
        scene["isect_offsets"], scene["flatten_ids"], softness=softness,
    )
    return out, inputs


def _loss(out, scene, channels):
    w_f, w_tf = random_weights(scene, channels, seed=1)
    w_b, w_tb = random_weights(scene, channels, seed=2)
    return ((out["front"] * w_f).sum() + (out["T_front"] * w_tf).sum()
            + (out["back"] * w_b).sum() + (out["T_back"] * w_tb).sum())


@pytest.mark.parametrize("channels", [3, 4])
def test_no_mesh_matches_gsplat(channels):
    """Without a mesh, the front layer is plain 3DGS (forward and gradients)."""
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
    # the mesh-related inputs get no gradient without a mesh
    assert our_in[4].grad.abs().max().item() == 0.0
    assert our_in[5].grad.abs().max().item() == 0.0


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("softness", [1.0, 0.3])
@pytest.mark.parametrize("coverage_grad", [False, True])
def test_matches_reference(channels, softness, coverage_grad):
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
    coverages = coverages.requires_grad_(coverage_grad)

    member = tile_membership(scene)
    ref_in = leaves(scene, INPUT_KEYS)
    ref_limits = limits.detach().clone().requires_grad_(True)
    ref = _reference(*ref_in, ref_limits, coverages.detach(), scene["Ks"], scene["depths"], member, W, H,
                     softness, skip_covered_back=not coverage_grad)
    _loss(ref, scene, channels).backward()

    out, our_in = _run(scene, limits, coverages, softness=softness)
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
    Moving the mesh surface across the Gaussians changes the image smoothly
    (no hard cut): small depth steps give small changes.
    """
    scene = random_scene(seed=4)
    W, H, C = scene["W"], scene["H"], scene["C"]
    coverages = torch.ones(C, H, W, device=DEVICE)
    inputs = [scene[k] for k in INPUT_KEYS]

    prev = None
    max_step = 0.0
    for d in torch.linspace(2.4, 3.6, 241).tolist():
        limits = torch.full((C, H, W), d, device=DEVICE)
        out, _ = _run(scene, limits, coverages, inputs=inputs)
        if prev is not None:
            max_step = max(max_step, (out["front"] - prev).abs().max().item())
        prev = out["front"]
    # 241 steps over the whole depth range: no single step should flip a Gaussian on/off
    assert max_step < 0.15, max_step


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
