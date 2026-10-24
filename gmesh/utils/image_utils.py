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
