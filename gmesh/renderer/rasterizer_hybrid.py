#
# Peizhi Yan
# Copyright (C) 2025
#

"""
Hybrid Gaussian / mesh compositing with a soft, per-ray depth test.

At pixel p (mesh depth D, mesh color c_m, mesh coverage a_m), each Gaussian is
split along the camera ray through p:

    f = Phi((D - z*) / (sigma * softness))

where z* is the depth of the Gaussian's density peak along the ray and sigma
its standard deviation along the ray (from the Gaussian's 3D covariance).
The part in front of the mesh (alpha * f) and the part behind it
(alpha * (1 - f)) are alpha-composited into two layers, which are combined as

    C = C_front + T_front * (a_m * c_m + (1 - a_m) * (C_back + T_back * bg))

so a Gaussian crossing the mesh surface is cut smoothly, following the surface,
and everything is differentiable, including w.r.t. the Gaussian depth and the
mesh depth.

The forward/backward CUDA kernels are in ./csrc/rasterize_hybrid.cu. The
projection, spherical harmonics and tile intersection stages are gsplat's
(differentiable) functions. Tested against gsplat v1.5.2.
"""

import math
from typing import Dict, Optional

import torch
from torch import Tensor

from gmesh.renderer._backend import load_extension
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    spherical_harmonics,
)


_SUPPORTED_CHANNELS = (1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33)


class _RasterizeHybrid(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,       # [C, N, 2]
        conics: Tensor,        # [C, N, 3]
        colors: Tensor,        # [C, N, D]
        opacities: Tensor,     # [C, N]
        mus: Tensor,           # [C, N, 3]
        precs: Tensor,         # [C, N, 6]
        depth_limits: Tensor,  # [C, H, W]
        coverages: Tensor,     # [C, H, W] (not differentiated here)
        Ks: Tensor,            # [C, 3, 3]
        softness: float,
        skip_covered_back: bool,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,    # [n_isects]
    ):
        out_front, T_front, out_back, T_back, last_ids = load_extension().rasterize_hybrid_fwd(
            means2d, conics, colors, opacities, mus, precs, depth_limits, coverages, Ks,
            softness, skip_covered_back, width, height, tile_size, isect_offsets, flatten_ids,
        )
        ctx.save_for_backward(
            means2d, conics, colors, opacities, mus, precs, depth_limits, coverages, Ks,
            isect_offsets, flatten_ids, T_front, T_back, last_ids,
        )
        ctx.softness = softness
        ctx.skip_covered_back = skip_covered_back
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.mark_non_differentiable(last_ids)
        return out_front, T_front, out_back, T_back, last_ids

    @staticmethod
    def backward(ctx, v_out_front, v_T_front, v_out_back, v_T_back, v_last_ids):
        (
            means2d, conics, colors, opacities, mus, precs, depth_limits, coverages, Ks,
            isect_offsets, flatten_ids, T_front, T_back, last_ids,
        ) = ctx.saved_tensors

        (
            v_means2d, v_conics, v_colors, v_opacities, v_mus, v_precs, v_depth_limits,
        ) = load_extension().rasterize_hybrid_bwd(
            means2d, conics, colors, opacities, mus, precs, depth_limits, coverages, Ks,
            ctx.softness, ctx.skip_covered_back, ctx.width, ctx.height, ctx.tile_size,
            isect_offsets, flatten_ids, T_front, T_back, last_ids,
            v_out_front.contiguous(), v_T_front.contiguous(),
            v_out_back.contiguous(), v_T_back.contiguous(),
        )
        return (
            v_means2d, v_conics, v_colors, v_opacities, v_mus, v_precs, v_depth_limits,
            None, None, None, None, None, None, None, None, None,
        )


def rasterize_to_pixels_hybrid(
    means2d: Tensor,       # [C, N, 2]
    conics: Tensor,        # [C, N, 3]
    colors: Tensor,        # [C, N, D]
    opacities: Tensor,     # [C, N]
    mus: Tensor,           # [C, N, 3] camera-space means
    precs: Tensor,         # [C, N, 6] camera-space precision (inverse covariance), upper triangle
    depth_limits: Tensor,  # [C, H, W] mesh depth, +inf where there is no mesh
    coverages: Tensor,     # [C, H, W] mesh coverage in [0, 1]
    Ks: Tensor,            # [C, 3, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,    # [n_isects]
    softness: float = 1.0,
) -> Dict[str, Tensor]:
    """
    Two-layer (in front of / behind the mesh) compositing of the Gaussians.

    The gradients w.r.t. `coverages` are not computed here (the caller composes
    the layers with the coverage in PyTorch). If `coverages` does not require
    grad, the back layer is skipped where the coverage is 1.

    Returns a dict with:
        front:   accumulated colors in front of the mesh. [C, H, W, D]
        T_front: transmittance of the front layer. [C, H, W, 1]
        back:    accumulated colors behind the mesh. [C, H, W, D]
        T_back:  transmittance of the back layer. [C, H, W, 1]
    """
    C, N = means2d.shape[:2]
    channels = colors.shape[-1]
    if channels > 33 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in _SUPPORTED_CHANNELS:
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat([colors, colors.new_zeros(*colors.shape[:-1], padded_channels)], dim=-1)
    else:
        padded_channels = 0

    skip_covered_back = not (torch.is_grad_enabled() and coverages.requires_grad)

    out_front, T_front, out_back, T_back, _ = _RasterizeHybrid.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        mus.contiguous(),
        precs.contiguous(),
        depth_limits.float().contiguous(),
        coverages.detach().float().contiguous(),
        Ks.float().contiguous(),
        float(softness),
        skip_covered_back,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    if padded_channels > 0:
        out_front = out_front[..., :channels]
        out_back = out_back[..., :channels]
    return {
        "front": out_front,
        "T_front": T_front[..., None],
        "back": out_back,
        "T_back": T_back[..., None],
    }


def quats_to_rotmats(quats: Tensor) -> Tensor:
    """quats: [N, 4] (w, x, y, z), not necessarily normalized. Returns [N, 3, 3]."""
    q = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).view(-1, 3, 3)


def camera_space_precisions(means: Tensor, quats: Tensor, scales: Tensor, viewmats: Tensor):
    """
    Camera-space means and precision matrices (inverse covariances) of 3D Gaussians.

    Returns:
        mus:   [C, N, 3]
        precs: [C, N, 6] upper triangle (p00, p01, p02, p11, p12, p22)
    """
    R_view = viewmats[:, :3, :3]  # [C, 3, 3]
    t_view = viewmats[:, :3, 3]   # [C, 3]
    mus = torch.einsum("cij,nj->cni", R_view, means) + t_view[:, None, :]

    R = quats_to_rotmats(quats)                            # [N, 3, 3]
    inv_s2 = 1.0 / scales.clamp(min=1e-8) ** 2             # [N, 3]
    P_world = (R * inv_s2[:, None, :]) @ R.transpose(-1, -2)  # R diag(1/s^2) R^T
    P_cam = torch.einsum("cij,njk,clk->cnil", R_view, P_world, R_view)  # R_v P R_v^T
    iu = torch.tensor([0, 0, 0, 1, 1, 2], device=means.device)
    ju = torch.tensor([0, 1, 2, 1, 2, 2], device=means.device)
    precs = P_cam[..., iu, ju]
    return mus, precs


def rasterization_hybrid(
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
    coverages: Tensor,     # [C, H, W]
    softness: float = 1.0,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    tile_size: int = 16,
    eps2d: float = 0.3,
    radius_clip: float = 0.0,
) -> Dict[str, Tensor]:
    """
    Mirrors gsplat.rendering.rasterization (non-packed, classic, pinhole,
    render_mode="RGB+D"), with the two-layer hybrid compositing step.

    Returns the dict of `rasterize_to_pixels_hybrid`, with channels
    [R, G, B, depth] (depth accumulated, not normalized).
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

    # 3D shape of the Gaussians in camera space, for the per-ray depth test
    mus, precs = camera_space_precisions(means, quats, scales, viewmats)

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

    return rasterize_to_pixels_hybrid(
        means2d,
        conics,
        colors,
        opacities,
        mus,
        precs,
        depth_limits,
        coverages,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        softness=softness,
    )


def composite_hybrid(
    layers: Dict[str, Tensor],
    mesh_rgb: Tensor,      # [C, H, W, 3] mesh color (not composited over the background)
    mesh_depth: Tensor,    # [C, H, W, 1]
    mesh_alpha: Tensor,    # [C, H, W, 1] mesh coverage
    bg_color,
):
    """
    Differentiable

    Combine the Gaussian layers (channels [R, G, B, depth]) with the mesh:

        C = C_front + T_front * (a_m * c_m + (1 - a_m) * (C_back + T_back * bg))

    Returns:
        rgb:   [C, H, W, 3]
        depth: [C, H, W, 1] expected depth over Gaussians and mesh
        alpha: [C, H, W, 1]
    """
    if not isinstance(bg_color, torch.Tensor):
        bg_color = torch.tensor(bg_color, dtype=mesh_rgb.dtype, device=mesh_rgb.device)
    bg_color = bg_color.view(1, 1, 1, 3)

    front, T_f = layers["front"], layers["T_front"]
    back, T_b = layers["back"], layers["T_back"]
    a_m = mesh_alpha

    behind_rgb = a_m * mesh_rgb + (1.0 - a_m) * (back[..., :3] + T_b * bg_color)
    rgb = front[..., :3] + T_f * behind_rgb
    alpha = 1.0 - T_f * (1.0 - a_m) * T_b
    depth_acc = front[..., 3:4] + T_f * (a_m * mesh_depth + (1.0 - a_m) * back[..., 3:4])
    depth = depth_acc / alpha.clamp(min=1e-10)
    return rgb, depth, alpha
