#
# Peizhi Yan
# Copyright (C) 2025
#

"""
Tests for the z-buffer (hard, center-depth) Gaussian rasterizer.

Requires a CUDA GPU and gsplat v1.5.2.
The first run JIT-compiles the CUDA extension, which takes a minute.

Run with:
    python -m pytest tests -v
or:
    python tests/test_zbuffer_rasterizer.py
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from gsplat.cuda._wrapper import rasterize_to_pixels

from gmesh.renderer.rasterizer_zbuffer import rasterize_to_pixels_zbuffer

from common import (
    DEVICE, assert_close, leaves, pixel_centers, random_scene, random_weights, tile_membership,
)


def _reference(means2d, conics, colors, opacities, depths, depth_limits, member, W, H):
    """
    Brute-force PyTorch version of the depth-limited rasterizer (autograd for backward).
    Mirrors the CUDA kernel: front-to-back order, alpha threshold, 0.999 clamp,
    T <= 1e-4 termination, and the z-test.
    """
    C, N = means2d.shape[:2]
    px, py = pixel_centers(W, H)

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


@pytest.mark.parametrize("channels", [3, 4, 6])
def test_infinite_depth_limits_match_gsplat(channels):
    """With no occluder, the result (and gradients) must equal gsplat's rasterizer."""
    scene = random_scene(channels=channels)
    W, H, C, ts = scene["W"], scene["H"], scene["C"], scene["tile_size"]
    limits = torch.full((C, H, W), float("inf"), device=DEVICE)
    w_c, w_a = random_weights(scene, channels)

    ref_in = leaves(scene)
    ref_c, ref_a = rasterize_to_pixels(
        *ref_in, W, H, ts, scene["isect_offsets"], scene["flatten_ids"], packed=False
    )
    ((ref_c * w_c).sum() + (ref_a * w_a).sum()).backward()

    our_in = leaves(scene)
    our_c, our_a = rasterize_to_pixels_zbuffer(
        *our_in, scene["depths"], limits, W, H, ts, scene["isect_offsets"], scene["flatten_ids"]
    )
    ((our_c * w_c).sum() + (our_a * w_a).sum()).backward()

    assert_close(our_c, ref_c, "colors", atol_scale=1e-6)
    assert_close(our_a, ref_a, "alphas", atol_scale=1e-6)
    for name, ours, ref in zip(("means2d", "conics", "colors", "opacities"), our_in, ref_in):
        assert_close(ours.grad, ref.grad, f"grad {name}", atol_scale=1e-5)


@pytest.mark.parametrize("channels", [3, 4])
def test_depth_limits_match_reference(channels):
    """With an occluder, compare forward and gradients to a brute-force reference."""
    scene = random_scene(channels=channels, seed=1)
    W, H, C, ts = scene["W"], scene["H"], scene["C"], scene["tile_size"]

    # a slanted occluder through the Gaussians, nothing in some pixels, and a
    # region in the very first tile where every Gaussian is occluded
    xs = torch.arange(W, device=DEVICE).float()
    limits = (2.6 + 0.8 * xs / W).view(1, 1, W).repeat(C, H, 1)
    limits[:, H // 2:, : W // 3] = float("inf")
    limits[:, :8, :8] = 0.5

    w_c, w_a = random_weights(scene, channels)
    member = tile_membership(scene)

    ref_in = leaves(scene)
    ref_c, ref_a = _reference(*ref_in, scene["depths"], limits, member, W, H)
    ((ref_c * w_c).sum() + (ref_a * w_a).sum()).backward()

    our_in = leaves(scene)
    our_c, our_a = rasterize_to_pixels_zbuffer(
        *our_in, scene["depths"], limits, W, H, ts, scene["isect_offsets"], scene["flatten_ids"]
    )
    ((our_c * w_c).sum() + (our_a * w_a).sum()).backward()

    # the occluder actually removed something
    assert (our_a.sum() < rasterize_to_pixels(*leaves(scene), W, H, ts, scene["isect_offsets"],
                                              scene["flatten_ids"], packed=False)[1].sum()).item()
    # fully occluded pixels are empty
    assert our_a[:, :8, :8].abs().max().item() == 0.0

    assert_close(our_c, ref_c, "colors")
    assert_close(our_a, ref_a, "alphas")
    for name, ours, ref in zip(("means2d", "conics", "colors", "opacities"), our_in, ref_in):
        assert_close(ours.grad, ref.grad, f"grad {name}")


def test_depth_bias():
    """depth_bias shifts the depth limit."""
    scene = random_scene(seed=2)
    W, H, C, ts = scene["W"], scene["H"], scene["C"], scene["tile_size"]
    args = (scene["means2d"], scene["conics"], scene["colors"], scene["opacities"], scene["depths"])
    limits = torch.full((C, H, W), 3.0, device=DEVICE)
    c1, a1 = rasterize_to_pixels_zbuffer(*args, limits + 0.25, W, H, ts, scene["isect_offsets"], scene["flatten_ids"])
    c2, a2 = rasterize_to_pixels_zbuffer(*args, limits, W, H, ts, scene["isect_offsets"], scene["flatten_ids"],
                                         depth_bias=0.25)
    torch.testing.assert_close(c1, c2)
    torch.testing.assert_close(a1, a2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
