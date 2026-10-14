#
# Peizhi Yan
# Copyright (C) 2025
#

"""
3DGS rasterization with a per-pixel depth limit (z-buffer).

Each Gaussian is depth-tested against a depth map (e.g. the mesh z-buffer)
during alpha compositing: at every pixel, only Gaussians in front of the
depth limit are composited. This makes Gaussians and mesh triangles occlude
each other correctly per pixel, instead of blending two finished images.

The forward pass is a custom CUDA kernel (./csrc), JIT-compiled on first use.
The backward pass is gsplat's own rasterization backward kernel, which is
exactly consistent with the depth-limited forward pass (see csrc for why).
The projection, spherical harmonics and tile intersection stages are gsplat's
regular (differentiable) functions.

Tested against gsplat v1.5.2, which this module relies on internally.
"""

import math
import os
from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import Tensor

from gsplat.cuda._wrapper import (
    _make_lazy_cuda_func,
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    spherical_harmonics,
)


_SUPPORTED_CHANNELS = (1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513)


@lru_cache(maxsize=None)
def _load_extension():
    from torch.utils.cpp_extension import load

    csrc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
    return load(
        name="gmesh_zbuffer_rasterizer",
        sources=[
            os.path.join(csrc, "ext.cpp"),
            os.path.join(csrc, "rasterize_zbuffer.cu"),
        ],
        extra_cflags=["-O3"],
        # same flags as gsplat, so alpha values match gsplat's backward kernel
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )


class _RasterizeToPixelsZBuffer(torch.autograd.Function):
    """Depth-limited forward (gmesh CUDA kernel), gsplat backward."""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,       # [C, N, 2]
        conics: Tensor,        # [C, N, 3]
        colors: Tensor,        # [C, N, D]
        opacities: Tensor,     # [C, N]
        depths: Tensor,        # [C, N]
        depth_limits: Tensor,  # [C, H, W]
        depth_bias: float,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,    # [n_isects]
    ) -> Tuple[Tensor, Tensor]:
        render_colors, render_alphas, last_ids = _load_extension().rasterize_to_pixels_zbuffer_fwd(
            means2d,
            conics,
            colors,
            opacities,
            depths,
            depth_limits,
            depth_bias,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        return render_colors, render_alphas

    @staticmethod
    def backward(ctx, v_render_colors: Tensor, v_render_alphas: Tensor):
        (
            means2d,
            conics,
            colors,
            opacities,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors

        _, v_means2d, v_conics, v_colors, v_opacities = _make_lazy_cuda_func(
            "rasterize_to_pixels_3dgs_bwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            None,  # backgrounds
            None,  # masks
            ctx.width,
            ctx.height,
            ctx.tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            False,  # absgrad
        )

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            None,  # depths (the z-test is a hard step: no gradient)
            None,  # depth_limits
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize_to_pixels_zbuffer(
    means2d: Tensor,       # [C, N, 2]
    conics: Tensor,        # [C, N, 3]
    colors: Tensor,        # [C, N, D]
    opacities: Tensor,     # [C, N]
    depths: Tensor,        # [C, N]
    depth_limits: Tensor,  # [C, H, W]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,    # [n_isects]
    depth_bias: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    Same as gsplat's `rasterize_to_pixels` (non-packed, no background), plus a
    per-pixel depth test: at pixel p, Gaussian i is composited only if
    depths[i] < depth_limits[p] + depth_bias.

    Args:
        depths:       Gaussian (center) depths, as returned by gsplat's projection.
        depth_limits: per-pixel depth limit, e.g. the mesh z-buffer, with +inf
                      where nothing occludes the Gaussians.
        depth_bias:   added to the depth limit; a small positive value avoids
                      z-fighting for Gaussians lying on a mesh surface.

    Returns:
        render_colors: accumulated colors, not composited with a background. [C, H, W, D]
        render_alphas: accumulated opacities. [C, H, W, 1]
    """
    C, N = means2d.shape[:2]
    assert means2d.shape == (C, N, 2), means2d.shape
    assert conics.shape == (C, N, 3), conics.shape
    assert colors.shape[:2] == (C, N), colors.shape
    assert opacities.shape == (C, N), opacities.shape
    assert depths.shape == (C, N), depths.shape
    assert depth_limits.shape == (C, image_height, image_width), depth_limits.shape

    # pad the channels to a number gsplat's backward kernel is compiled for
    channels = colors.shape[-1]
    if channels > 513 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in _SUPPORTED_CHANNELS:
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat([colors, colors.new_zeros(*colors.shape[:-1], padded_channels)], dim=-1)
    else:
        padded_channels = 0

    # Prepend one dummy intersection that no tile reads, so that every tile
    # range starts at index >= 1. The kernel then marks "nothing contributed"
    # with last_ids = range_start - 1 >= 0, which gsplat's backward handles
    # correctly (a negative value would not be).
    flatten_ids = torch.cat([flatten_ids.new_zeros(1), flatten_ids])
    isect_offsets = isect_offsets + 1

    render_colors, render_alphas = _RasterizeToPixelsZBuffer.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        depths.detach().float().contiguous(),
        depth_limits.detach().float().contiguous(),
        float(depth_bias),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    if padded_channels > 0:
        render_colors = render_colors[..., :channels]
    return render_colors, render_alphas


def rasterization_zbuffer(
    means: Tensor,         # [N, 3]
    quats: Tensor,         # [N, 4]
    scales: Tensor,        # [N, 3]
    opacities: Tensor,     # [N]
    colors: Tensor,        # [N, K, 3] SH coefficients
    sh_degree: int,
    viewmats: Tensor,      # [C, 4, 4]
    Ks: Tensor,            # [C, 3, 3]
    width: int,
    height: int,
    depth_limits: Tensor,  # [C, H, W]
    depth_bias: float = 0.0,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    tile_size: int = 16,
    eps2d: float = 0.3,
    radius_clip: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Mirrors gsplat.rendering.rasterization (non-packed, classic, pinhole,
    render_mode="RGB+D"), with the depth-limited rasterization step.

    Returns:
        rgb:       accumulated colors (premultiplied, no background). [C, H, W, 3]
        depth_acc: accumulated depth sum_i(w_i * z_i), not normalized. [C, H, W, 1]
        alpha:     accumulated opacity. [C, H, W, 1]
    """
    N = means.shape[0]
    C = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3, colors.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # project Gaussians to 2D
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=False,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        calc_compensations=False,
        opacities=opacities,  # tighter bound for radii
    )
    opacities = opacities.repeat(C, 1)  # [C, N]

    # view-dependent colors from SH
    campos = torch.inverse(viewmats)[:, :3, 3]  # [C, 3]
    dirs = means[None, :, :] - campos[:, None, :]  # [C, N, 3]
    masks = (radii > 0).all(dim=-1)  # [C, N]
    shs = colors.expand(C, -1, -1, -1)  # [C, N, K, 3]
    colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
    colors = torch.clamp_min(colors + 0.5, 0.0)

    # render depth as an extra channel (as gsplat's "RGB+D")
    colors = torch.cat([colors, depths[..., None]], dim=-1)  # [C, N, 4]

    # tile intersections, sorted front-to-back within each tile
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_cameras=C,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    render_colors, render_alphas = rasterize_to_pixels_zbuffer(
        means2d,
        conics,
        colors,
        opacities,
        depths,
        depth_limits,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        depth_bias=depth_bias,
    )

    rgb = render_colors[..., :3]
    depth_acc = render_colors[..., 3:4]
    return rgb, depth_acc, render_alphas
