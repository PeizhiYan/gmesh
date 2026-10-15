#
# Peizhi Yan
# Copyright (C) 2025
#

"""
Hybrid Gaussian / mesh compositing with a soft, per-pixel depth test.

At pixel p (mesh depth D, mesh color c_m, mesh coverage a_m), each Gaussian is
split at the mesh depth:

    f = Phi((D - z*(p)) / (sigma * softness))

where z*(p) and sigma are the mean and standard deviation of the Gaussian's
depth at pixel p, in the same local affine (EWA) approximation as its 2D splat:
the camera-space depth conditioned on the projected position,

    z*(p) = z_c + g . (p - mean2d),   g = (Sigma_uv + eps I)^-1 Sigma_uv,z
    sigma^2 = Sigma_zz - Sigma_z,uv g

so a flat Gaussian lying on a tilted surface has a depth that tilts with it.
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
        planes: Tensor,        # [C, N, 4] (z_c, g_x, g_y, sigma)
        depth_limits: Tensor,  # [C, H, W]
        coverages: Tensor,     # [C, H, W] (not differentiated here)
        softness: float,
        skip_covered_back: bool,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,    # [n_isects]
    ):
        out_front, T_front, out_back, T_back, last_ids = load_extension().rasterize_hybrid_fwd(
            means2d, conics, colors, opacities, planes, depth_limits, coverages,
            softness, skip_covered_back, width, height, tile_size, isect_offsets, flatten_ids,
        )
        ctx.save_for_backward(
            means2d, conics, colors, opacities, planes, depth_limits, coverages,
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
            means2d, conics, colors, opacities, planes, depth_limits, coverages,
            isect_offsets, flatten_ids, T_front, T_back, last_ids,
        ) = ctx.saved_tensors

        (
            v_means2d, v_conics, v_colors, v_opacities, v_planes, v_depth_limits,
        ) = load_extension().rasterize_hybrid_bwd(
            means2d, conics, colors, opacities, planes, depth_limits, coverages,
            ctx.softness, ctx.skip_covered_back, ctx.width, ctx.height, ctx.tile_size,
            isect_offsets, flatten_ids, T_front, T_back, last_ids,
            v_out_front.contiguous(), v_T_front.contiguous(),
            v_out_back.contiguous(), v_T_back.contiguous(),
        )
        return (
            v_means2d, v_conics, v_colors, v_opacities, v_planes, v_depth_limits,
            None, None, None, None, None, None, None, None,
        )


def rasterize_to_pixels_hybrid(
    means2d: Tensor,       # [C, N, 2]
    conics: Tensor,        # [C, N, 3]
    colors: Tensor,        # [C, N, D]
    opacities: Tensor,     # [C, N]
    planes: Tensor,        # [C, N, 4] per-pixel depth model (z_c, g_x, g_y, sigma), see gaussian_depth_planes
    depth_limits: Tensor,  # [C, H, W] mesh depth, +inf where there is no mesh
    coverages: Tensor,     # [C, H, W] mesh coverage in [0, 1]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,    # [n_isects]
    softness: float = 1.0,
    skip_covered_back: bool = True,
) -> Dict[str, Tensor]:
    """
    Two-layer (in front of / behind the mesh) compositing of the Gaussians.

    The gradients w.r.t. `coverages` are not computed here (the caller composes
    the layers with the coverage in PyTorch). With `skip_covered_back`, the back
    layer is not computed where the coverage is 1 (it is multiplied by
    1 - coverage = 0 there). This is exact as long as the coverage has no
    gradient at those pixels, which holds for the mesh rasterizer (coverage is
    constant 1 inside the mesh, also with antialiasing).

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

    out_front, T_front, out_back, T_back, _ = _RasterizeHybrid.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        planes.contiguous(),
        depth_limits.float().contiguous(),
        coverages.detach().float().contiguous(),
        float(softness),
        bool(skip_covered_back),
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


def gaussian_depth_planes(
    means: Tensor,     # [N, 3]
    quats: Tensor,     # [N, 4]
    scales: Tensor,    # [N, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,        # [C, 3, 3]
    eps2d: float = 0.3,
) -> Tensor:
    """
    Per-pixel depth model of each Gaussian, in the local affine (EWA)
    approximation used for its 2D splat. With the Jacobian J of the projection
    at the camera-space mean and the camera-space covariance Sigma, the joint
    covariance of the projected position (u, v) and the depth z is
    [[J Sigma J^T, J Sigma e_z], [e_z^T Sigma J^T, Sigma_zz]]. The depth
    conditioned on (u, v) = p is Gaussian with

        mean  z_c + g . (p - mean2d),  g = (J Sigma J^T + eps2d I)^-1 J Sigma e_z
        var   Sigma_zz - (J Sigma e_z) . g

    (eps2d is the same low-pass dilation gsplat adds to the 2D covariance).
    Note |g . (p - mean2d)| <= r * sqrt(Sigma_zz) for a pixel at Mahalanobis
    distance r from the 2D splat, so the depth stays within the Gaussian's extent.

    Returns:
        planes: [C, N, 4] (z_c, g_x, g_y, sigma)
    """
    R_view = viewmats[:, :3, :3]  # [C, 3, 3]
    t_view = viewmats[:, :3, 3]   # [C, 3]
    mus = torch.einsum("cij,nj->cni", R_view, means) + t_view[:, None, :]  # [C, N, 3]

    R = quats_to_rotmats(quats)                                    # [N, 3, 3]
    M = R * scales[:, None, :]                                     # R S
    Sigma_world = M @ M.transpose(-1, -2)                          # [N, 3, 3]
    Sigma = torch.einsum("cij,njk,clk->cnil", R_view, Sigma_world, R_view)  # [C, N, 3, 3]

    x, y = mus[..., 0], mus[..., 1]
    z = mus[..., 2].clamp(min=1e-6)
    fx = Ks[:, 0, 0][:, None]
    fy = Ks[:, 1, 1][:, None]
    zeros = torch.zeros_like(z)
    J = torch.stack([
        torch.stack([fx / z, zeros, -fx * x / (z * z)], dim=-1),
        torch.stack([zeros, fy / z, -fy * y / (z * z)], dim=-1),
    ], dim=-2)                                                     # [C, N, 2, 3]

    cov2d = J @ Sigma @ J.transpose(-1, -2)                        # [C, N, 2, 2]
    cov2d = cov2d + eps2d * torch.eye(2, device=means.device, dtype=means.dtype)
    cross = (J @ Sigma[..., :, 2:3])[..., 0]                       # [C, N, 2] Cov(uv, z)
    a, b, d = cov2d[..., 0, 0], cov2d[..., 0, 1], cov2d[..., 1, 1]
    det = (a * d - b * b).clamp(min=1e-12)
    g = torch.stack([d * cross[..., 0] - b * cross[..., 1],
                     a * cross[..., 1] - b * cross[..., 0]], dim=-1) / det[..., None]  # cov2d^-1 cross
    var = Sigma[..., 2, 2] - (cross * g).sum(dim=-1)
    sigma = var.clamp(min=1e-12).sqrt()
    return torch.stack([mus[..., 2], g[..., 0], g[..., 1], sigma], dim=-1).contiguous()


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
    cull_occluded: bool = True,
) -> Dict[str, Tensor]:
    """
    Mirrors gsplat.rendering.rasterization (non-packed, classic, pinhole,
    render_mode="RGB+D"), with the two-layer hybrid compositing step.

    With `cull_occluded`, Gaussians that are certainly hidden behind the mesh in
    a fully covered tile are removed from that tile's list before compositing
    (they would contribute nothing; see `cull_occluded_intersections`).

    Returns the dict of `rasterize_to_pixels_hybrid`, with channels
    [R, G, B, depth] (depth accumulated, not normalized), plus the number of
    tile intersections before ('n_isects') and after ('n_isects_rendered') culling.
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

    # per-pixel depth model of the Gaussians, for the soft depth test
    planes = gaussian_depth_planes(means, quats, scales, viewmats, Ks, eps2d=eps2d)

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

    n_isects = flatten_ids.numel()
    if cull_occluded:
        isect_offsets, flatten_ids = cull_occluded_intersections(
            isect_offsets, flatten_ids, depths, scales, depth_limits, coverages, tile_size, softness
        )

    layers = rasterize_to_pixels_hybrid(
        means2d,
        conics,
        colors,
        opacities,
        planes,
        depth_limits,
        coverages,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        softness=softness,
    )
    layers["n_isects"] = n_isects
    layers["n_isects_rendered"] = flatten_ids.numel()
    return layers


@torch.no_grad()
def cull_occluded_intersections(
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,    # [n_isects], sorted by tile then depth
    depths: Tensor,         # [C, N] Gaussian center depths
    scales: Tensor,         # [N, 3]
    depth_limits: Tensor,   # [C, H, W] mesh depth, +inf where there is no mesh
    coverages: Tensor,      # [C, H, W]
    tile_size: int,
    softness: float,
):
    """
    Remove Gaussians that cannot contribute in tiles fully covered by the mesh.

    In a tile where every pixel has coverage 1 (no back layer) and the mesh is
    at most at depth D_max, a Gaussian contributes only through its front part
    alpha * Phi((D - z*) / (sigma * softness)). With s_max its largest scale,
    sigma <= sqrt(Sigma_zz) <= s_max, and z* >= z_center - r * s_max at every
    pixel where alpha >= 1/255, where r <= sqrt(2 ln 255) ~ 3.33 is the
    Mahalanobis radius of its 2D footprint (see gaussian_depth_planes; we use 4).
    If z_center - 4 s_max >= D_max + 3 s_max * softness, the front part stays
    below alpha * Phi(-3) < 1/255 at every pixel of the tile, i.e. it is skipped
    by the compositing anyway, so removing it does not change the result.
    """
    C, tile_height, tile_width = isect_offsets.shape
    H, W = depth_limits.shape[1:]
    n_tiles = C * tile_height * tile_width
    n_isects = flatten_ids.numel()
    if n_isects == 0:
        return isect_offsets, flatten_ids
    device = flatten_ids.device

    # per tile: fully covered, and the farthest mesh depth
    pad_h, pad_w = tile_height * tile_size - H, tile_width * tile_size - W
    d = torch.nn.functional.pad(depth_limits.float(), (0, pad_w, 0, pad_h), value=-float("inf"))
    cov = torch.nn.functional.pad(coverages.float(), (0, pad_w, 0, pad_h), value=1.0)
    d_max = d.view(C, tile_height, tile_size, tile_width, tile_size).amax(dim=(2, 4)).flatten()
    full = cov.view(C, tile_height, tile_size, tile_width, tile_size).amin(dim=(2, 4)).flatten() >= 1.0
    full = full & torch.isfinite(d_max)
    if not bool(full.any()):
        return isect_offsets, flatten_ids

    offsets = isect_offsets.flatten().long()
    counts = torch.diff(torch.cat([offsets, offsets.new_tensor([n_isects])]))
    tile_of = torch.repeat_interleave(torch.arange(n_tiles, device=device), counts)

    g = flatten_ids.long()                                        # index into [C * N]
    z = depths.detach().flatten()[g]
    s = scales.detach().abs().amax(dim=-1).repeat(C)[g]
    hidden = full[tile_of] & (z - 4.0 * s >= d_max[tile_of] + 3.0 * softness * s)
    if not bool(hidden.any()):
        return isect_offsets, flatten_ids

    keep = ~hidden
    new_counts = torch.bincount(tile_of[keep], minlength=n_tiles)
    new_offsets = (torch.cumsum(new_counts, dim=0) - new_counts).to(torch.int32)
    return new_offsets.view(C, tile_height, tile_width).contiguous(), flatten_ids[keep].contiguous()


def composite_hybrid(
    layers: Dict[str, Tensor],
    mesh_color: Tensor,    # [C, H, W, 3] mesh color premultiplied by the coverage (a_m * c_m)
    mesh_depth: Tensor,    # [C, H, W, 1]
    mesh_alpha: Tensor,    # [C, H, W, 1] mesh coverage a_m
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
        bg_color = torch.tensor(bg_color, dtype=mesh_color.dtype, device=mesh_color.device)
    bg_color = bg_color.view(1, 1, 1, 3)

    front, T_f = layers["front"], layers["T_front"]
    back, T_b = layers["back"], layers["T_back"]
    a_m = mesh_alpha

    behind_rgb = mesh_color + (1.0 - a_m) * (back[..., :3] + T_b * bg_color)
    rgb = front[..., :3] + T_f * behind_rgb
    alpha = 1.0 - T_f * (1.0 - a_m) * T_b
    depth_acc = front[..., 3:4] + T_f * (a_m * mesh_depth + (1.0 - a_m) * back[..., 3:4])
    depth = depth_acc / alpha.clamp(min=1e-10)
    return rgb, depth, alpha
