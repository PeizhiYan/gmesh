#
# Peizhi Yan
# Copyright (C) 2025
#

import torch

def spherical_harmonics_degree(S: int) -> int:
    """
    Given number of spherical harmonics coefficients S, return the degree l.
    Raises ValueError if S is not a perfect square.
    """
    l = int(S ** 0.5) - 1
    if (l + 1) ** 2 != S:
        raise ValueError(f"S={S} is not a valid number of spherical harmonics coefficients (must be a perfect square).")
    return l

class Gaussians:
    def __init__(self, 
                 means : torch.tensor, 
                 quats : torch.tensor,
                 scales : torch.tensor,
                 opacities : torch.tensor,
                 colors : torch.tensor,
                 allow_check : bool = True,
                 ):
        """
        All inputs should be torch tensors (with gradients if needed).

        Args:
            means:     [N, 3]    torch.Tensor
            quats:     [N, 4]    torch.Tensor
            scales:    [N, 3]    torch.Tensor
            opacities: [N]       torch.Tensor
            colors:    [N, S, 3] torch.Tensor
            allow_check: whether to validate the input tensors. Boolean
        
        Notes:
            N is the number of Gaussian points
            S is the number of Spherical Harmonics
        """
        # meta data
        self.num_gaussians = means.shape[0]
        self.num_sh = colors.shape[1]
        self.sh_degree = spherical_harmonics_degree(self.num_sh)
        self.device = means.device

        if allow_check:
            # validate tensor shapes
            assert means.shape == (self.num_gaussians, 3), \
                f"means should have shape ({self.num_gaussians}, 3), but got {means.shape}"
            assert quats.shape == (self.num_gaussians, 4), \
                f"quats should have shape ({self.num_gaussians}, 4), but got {quats.shape}"
            assert scales.shape == (self.num_gaussians, 3), \
                f"scales should have shape ({self.num_gaussians}, 3), but got {scales.shape}"
            assert opacities.shape == (self.num_gaussians,), \
                f"opacities should have shape ({self.num_gaussians},), but got {opacities.shape}"
            assert colors.shape == (self.num_gaussians, self.num_sh, 3), \
                f"colors should have shape ({self.num_sh}, {self.num_gaussians}, 3), but got {colors.shape}"
            
            # device check
            devices = {tensor.device for tensor in [means, quats, scales, opacities, colors]}
            if len(devices) != 1:
                raise ValueError(
                    f"All tensors must be on the same device, but found devices: {devices}"
                )

        # 3D Gaussians data
        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.colors = colors


    def __repr__(self):
        return (f"Gaussians(N={self.num_gaussians}, SH Degrees={self.sh_degree}, "
                f"means={self.means.shape}, quats={self.quats.shape}, "
                f"scales={self.scales.shape}, opacities={self.opacities.shape}, "
                f"colors={self.colors.shape})")
    
