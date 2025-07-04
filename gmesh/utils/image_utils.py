#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
# import torch.nn.functional as F
# from math import exp
# import numpy as np
import matplotlib.pyplot as plt



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
        epsilon: depth offset for 3DGS

    Returns:
        rgb_out:   [B, H, W, 3]
        depth_out: [B, H, W, 1]
        alpha_out: [B, H, W, 1]

    Note:
        Because the depth map of 3DGS is approximate, may not reflect the real depth at
        some pixels
    """

    # mask: True where Gaussian is in front
    gaussian_in_front = (depth_g < depth_m)

    # plt.imshow(gaussian_in_front.detach().cpu()[0,...])
    # plt.show()

    # foreground selection
    C_fg = torch.where(gaussian_in_front, rgb_g, rgb_m)
    A_fg = torch.where(gaussian_in_front, alpha_g, alpha_m)
    D_fg = torch.where(gaussian_in_front, depth_g, depth_m)

    # background selection
    C_bg = torch.where(gaussian_in_front, rgb_m, rgb_g)
    A_bg = torch.where(gaussian_in_front, alpha_m, alpha_g)
    D_bg = torch.where(gaussian_in_front, depth_m, depth_g)


    # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # # Assume batch size B >= 1, show the first sample [0]
    # axs[0].imshow(C_fg[0].detach().cpu().numpy())
    # axs[0].set_title('C_fg (RGB)')
    # axs[0].axis('off')
    # axs[1].imshow(A_fg[0, ..., 0].detach().cpu().numpy(), cmap='gray')
    # axs[1].set_title('A_fg (Alpha)')
    # axs[1].axis('off')
    # axs[2].imshow(D_fg[0, ..., 0].detach().cpu().numpy(), cmap='viridis')
    # axs[2].set_title('D_fg (Depth)')
    # axs[2].axis('off')
    # plt.tight_layout()
    # plt.show()

    # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # # Assume batch size B >= 1, show the first sample [0]
    # axs[0].imshow(C_bg[0].detach().cpu().numpy())
    # axs[0].set_title('C_bg (RGB)')
    # axs[0].axis('off')
    # axs[1].imshow(A_bg[0, ..., 0].detach().cpu().numpy(), cmap='gray')
    # axs[1].set_title('A_bg (Alpha)')
    # axs[1].axis('off')
    # axs[2].imshow(D_bg[0, ..., 0].detach().cpu().numpy(), cmap='viridis')
    # axs[2].set_title('D_bg (Depth)')
    # axs[2].axis('off')
    # plt.tight_layout()
    # plt.show()


    # alpha compositing
    rgb_blend = C_fg * A_fg + C_bg * (1 - A_fg)
    alpha_blend = A_fg + A_bg * (1 - A_fg)

    # for the output depth, take foreground where alpha_fg > 0, otherwise background
    depth_blend = torch.where(A_fg > 0, D_fg, D_bg)

    return rgb_blend, depth_blend, alpha_blend




