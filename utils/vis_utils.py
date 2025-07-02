#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
import numpy as np
import matplotlib.pyplot as plt

def normalize_depth(depth):
    depth_np = depth.detach().cpu().numpy()[0,...,0]   # [H,W]
    # Normalize to 0-1, then 3 channels
    depth_pos = depth_np[depth_np > 0.1]
    if depth_pos.size > 0:
        dmin = np.percentile(depth_pos, 5)
        dmax = np.percentile(depth_pos, 95)
    else:
        dmin, dmax = 0.0, 1.0  # fallback if all are zero
    # dmin, dmax = np.percentile(depth_np, 1), np.percentile(depth_np, 99)
    dnorm = (depth_np - dmin) / max(1e-6, dmax - dmin)
    dnorm = np.clip(dnorm, 0, 1)
    return dnorm # np.array [H,W]

def visualize_rgbd_alpha(rgb : torch.Tensor, depth : torch.Tensor, alpha : torch.Tensor, 
                         norm_depth : bool = True):
    rgb = np.clip(rgb.detach().cpu().numpy(), 0.0, 1.0)[0,...]
    alpha = alpha.detach().cpu().numpy()[0,...,0]
    if norm_depth:
        depth = normalize_depth(depth)
    else:
        depth = depth.detach().cpu().numpy()[0,...,0]

    fig, axes = plt.subplots(1, 3, figsize=(15,3), constrained_layout=True)

    axes[0].imshow(rgb)
    axes[0].set_title('RGB')
    axes[0].axis('off')

    im1 = axes[1].imshow(depth, cmap='magma')
    if norm_depth:
        axes[1].set_title('Depth (Normalized)')
    else:
        axes[1].set_title('Depth')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(alpha, cmap='gray')
    axes[2].set_title('Alpha')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.show()

