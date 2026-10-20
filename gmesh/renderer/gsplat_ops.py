#
# Adapted from gsplat v1.5.2 (https://github.com/nerfstudio-project/gsplat),
# file gsplat/cuda/_wrapper.py (spherical_harmonics, fully_fused_projection,
# isect_tiles, isect_offset_encode, _FullyFusedProjection, _SphericalHarmonics).
# Copyright the gsplat authors. Licensed under the Apache License, Version 2.0
# (see gmesh/renderer/csrc/gsplat/LICENSE).
#
# Modified for Gmesh (Peizhi Yan, 2026):
#   - calls the kernels ported into Gmesh's own CUDA extension (./csrc/gsplat)
#   - kept only the pinhole camera, quaternion + scale parameterization,
#     non-packed mode, without opacity compensation
#

"""
Differentiable projection, tile intersection and spherical harmonics of 3D
Gaussians, the stages before Gmesh's hybrid compositing. The CUDA kernels are
ported from gsplat v1.5.2 and give the same results.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from gmesh.renderer._backend import load_extension


class _FullyFusedProjection(torch.autograd.Function):
    """Projects Gaussians to 2D (EWA splatting, pinhole camera)."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,      # [N, 3]
        quats: Tensor,      # [N, 4]
        scales: Tensor,     # [N, 3]
        opacities: Tensor,  # [N]
        viewmats: Tensor,   # [C, 4, 4]
        Ks: Tensor,         # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        radii, means2d, depths, conics = load_extension().projection_fwd(
            means, quats, scales, opacities, viewmats, Ks,
            width, height, eps2d, near_plane, far_plane, radius_clip,
        )
        ctx.save_for_backward(means, quats, scales, viewmats, Ks, radii, conics)
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.mark_non_differentiable(radii)
        return radii, means2d, depths, conics

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics):
        means, quats, scales, viewmats, Ks, radii, conics = ctx.saved_tensors
        v_means, v_quats, v_scales, v_viewmats = load_extension().projection_bwd(
            means, quats, scales, viewmats, Ks,
            ctx.width, ctx.height, ctx.eps2d,
            radii, conics,
            v_means2d.contiguous(), v_depths.contiguous(), v_conics.contiguous(),
            ctx.needs_input_grad[4],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_quats = None
        if not ctx.needs_input_grad[2]:
            v_scales = None
        if not ctx.needs_input_grad[4]:
            v_viewmats = None
        # the opacities only select the radii (no gradient)
        return v_means, v_quats, v_scales, None, v_viewmats, None, None, None, None, None, None, None


def fully_fused_projection(
    means: Tensor,      # [N, 3]
    quats: Tensor,      # [N, 4] (need not be normalized)
    scales: Tensor,     # [N, 3]
    opacities: Tensor,  # [N], used for an opacity-aware bounding box
    viewmats: Tensor,   # [C, 4, 4] world-to-camera
    Ks: Tensor,         # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Projects 3D Gaussians to 2D (same as gsplat's fully_fused_projection with
    packed=False, camera_model="pinhole", calc_compensations=False).

    Gaussians outside the frustum, or too small / transparent, get radii 0 and
    are ignored by the later stages.

    Returns:
        radii:   [C, N, 2] int32 bounding-box radii in pixels (0: culled)
        means2d: [C, N, 2] projected means
        depths:  [C, N] camera-space depths
        conics:  [C, N, 3] inverse of the 2D covariances (upper triangle)
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert quats.size() == (N, 4), quats.size()
    assert scales.size() == (N, 3), scales.size()
    assert opacities.size() == (N,), opacities.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()
    return _FullyFusedProjection.apply(
        means.contiguous(), quats.contiguous(), scales.contiguous(), opacities.contiguous(),
        viewmats.contiguous(), Ks.contiguous(),
        int(width), int(height), float(eps2d), float(near_plane), float(far_plane), float(radius_clip),
    )


class _SphericalHarmonics(torch.autograd.Function):
    """Spherical Harmonics"""

    @staticmethod
    def forward(ctx, sh_degree: int, dirs: Tensor, coeffs: Tensor, masks: Optional[Tensor]) -> Tensor:
        colors = load_extension().sh_fwd(sh_degree, dirs, coeffs, masks)
        ctx.save_for_backward(dirs, coeffs, masks)
        ctx.sh_degree = sh_degree
        ctx.num_bases = coeffs.shape[-2]
        return colors

    @staticmethod
    def backward(ctx, v_colors: Tensor):
        dirs, coeffs, masks = ctx.saved_tensors
        compute_v_dirs = ctx.needs_input_grad[1]
        v_coeffs, v_dirs = load_extension().sh_bwd(
            ctx.num_bases, ctx.sh_degree, dirs, coeffs, masks, v_colors.contiguous(), compute_v_dirs
        )
        if not compute_v_dirs:
            v_dirs = None
        return None, v_dirs, v_coeffs, None


def spherical_harmonics(
    degrees_to_use: int,
    dirs: Tensor,    # [..., 3]
    coeffs: Tensor,  # [..., K, 3]
    masks: Optional[Tensor] = None,  # [...] optional, skip some computation
) -> Tensor:
    """
    Evaluates spherical harmonics (degree <= 3) in the given directions. [..., 3]
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    assert dirs.shape[:-1] == coeffs.shape[:-2], (dirs.shape, coeffs.shape)
    assert dirs.shape[-1] == 3, dirs.shape
    assert coeffs.shape[-1] == 3, coeffs.shape
    if masks is not None:
        assert masks.shape == dirs.shape[:-1], masks.shape
        masks = masks.contiguous()
    return _SphericalHarmonics.apply(degrees_to_use, dirs.contiguous(), coeffs.contiguous(), masks)


@torch.no_grad()
def isect_tiles(
    means2d: Tensor,  # [C, N, 2]
    radii: Tensor,    # [C, N, 2]
    depths: Tensor,   # [C, N]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Maps projected Gaussians to the image tiles they overlap.

    Returns:
        tiles_per_gauss: [C, N] int32 number of tiles per Gaussian
        isect_ids:       [n_isects] int64 camera id | tile id | depth, sorted if `sort`
        flatten_ids:     [n_isects] int32 index into [C * N]
    """
    C, N, _ = means2d.shape
    assert means2d.shape == (C, N, 2), means2d.size()
    assert radii.shape == (C, N, 2), radii.size()
    assert depths.shape == (C, N), depths.size()
    return load_extension().isect_tiles(
        means2d.contiguous(), radii.contiguous(), depths.contiguous(),
        int(tile_size), int(tile_width), int(tile_height), bool(sort),
    )


@torch.no_grad()
def isect_offset_encode(isect_ids: Tensor, n_cameras: int, tile_width: int, tile_height: int) -> Tensor:
    """Start offset of each tile in the sorted intersections. [C, tile_height, tile_width] int32"""
    return load_extension().isect_offset_encode(
        isect_ids.contiguous(), int(n_cameras), int(tile_width), int(tile_height)
    )
