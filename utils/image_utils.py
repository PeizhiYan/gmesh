#
# Peizhi Yan
# Copyright (C) 2025
#

import torch


def composite_with_bg(rgb, alpha, bg_color):
    """
    Differentiable

    Args:
        rgb:   [B, H, W, 3], values in [0,1]
        alpha: [B, H, W, 1], values in [0,1]
        bg_color: (3,) or (1,3) tuple/list/torch.Tensor, e.g., (1.0, 1.0, 1.0)

    Returns:
        rgb_composited: [B, H, W, 3]
    """
    with torch.no_grad():
        # Convert bg_color to tensor on the correct device
        if not isinstance(bg_color, torch.Tensor):
            bg_color = torch.tensor(bg_color, dtype=rgb.dtype, device=rgb.device)
        bg_color = bg_color.view(1, 1, 1, 3)  # [1,1,1,3] to broadcast

    # Alpha composite over background
    return rgb * alpha + bg_color * (1 - alpha)


def blend_rgba(rgb_m, depth_m, alpha_m, rgb_g, depth_g, alpha_g):
    """
    Differentiable

    Blends mesh and Gaussian renders per pixel, compositing the closer one in front.

    Args:
        rgb_m, rgb_g:       [B, H, W, 3]
        depth_m, depth_g:   [B, H, W, 1]
        alpha_m, alpha_g:   [B, H, W, 1]

    Returns:
        rgb_out:   [B, H, W, 3]
        depth_out: [B, H, W, 1]
        alpha_out: [B, H, W, 1]
    """
    # mask: True where mesh is in front, False where Gaussian is in front or equal
    mesh_in_front = (depth_m < depth_g)

    # foreground and background selection
    C_fg = torch.where(mesh_in_front, rgb_m, rgb_g)
    A_fg = torch.where(mesh_in_front, alpha_m, alpha_g)
    D_fg = torch.where(mesh_in_front, depth_m, depth_g)
    C_bg = torch.where(mesh_in_front, rgb_g, rgb_m)
    A_bg = torch.where(mesh_in_front, alpha_g, alpha_m)
    D_bg = torch.where(mesh_in_front, depth_g, depth_m)

    # alpha compositing ("over" operator)
    rgb_blend = C_fg * A_fg + C_bg * (1 - A_fg)
    alpha_blend = A_fg + A_bg * (1 - A_fg)

    # for the output depth, take foreground where alpha_fg > 0, otherwise background
    depth_blend = torch.where(A_fg > 0, D_fg, D_bg)

    return rgb_blend, depth_blend, alpha_blend


