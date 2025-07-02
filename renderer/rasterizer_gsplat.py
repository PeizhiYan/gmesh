#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
from gsplat.rendering import rasterization

from scenes.gaussians import Gaussians
from scenes.cameras import PerspectiveCamera
# from utils.graphics_utils import build_view_matrix
from utils.image_utils import composite_with_bg

class GaussianRasterizer:

    def __init__(self, camera : PerspectiveCamera, tile_size = 16, bg_color = (1.0,1.0,1.0)):
        self.camera = camera
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.z_near = camera.z_near
        self.z_far = camera.z_far
        self.tile_size = tile_size      # tile size, as used in 3DGS
        self.bg_color = bg_color        # background color

    def rasterize(self,
                  gaussians : Gaussians,
                  camera_pose : torch.Tensor,
                  ):
        """
        Rasterize 3D Gaussians from multiple camera views.

        Args:
            gaussians (Gaussians): Contains means, quats, scales, opacities, colors, sh_degree.
            camera_pose (torch.Tensor): 
                - for PerspectiveCamera, [B,6] (yaw, pitch, roll, dx, dy, dz);
                - for OrbitCamera, [B, 3] (elevation, azimuth, distance).

        Returns:
            rgb (torch.Tensor):   [B, H, W, 3]
            depth (torch.Tensor): [B, H, W, 1]
            alpha (torch.Tensor): [B, H, W, 1]

        Notes:
            B is batch size (number of camera views to render)
            N is the number of Gaussians
            S is the number of SH channels
        """
        batch_size = camera_pose.shape[0]
        device = gaussians.device
        bg_tensor = torch.tensor(self.bg_color).view(1, 3).repeat(batch_size, 1).to(device) # [B, 3]

        # get intrinsic matrix
        Ks = self.camera.get_intrinsic_matrix(batch_size=batch_size) # [B, 3, 3]

        # build view matrix
        viewmats = self.camera.get_view_matrix(camera_pose)

        # rasterize through gsplat
        render_colors, render_alphas, meta = rasterization(
            means = gaussians.means,            # [N, 3]
            quats = gaussians.quats,            # [N, 4]
            scales = gaussians.scales,          # [N, 3]
            opacities = gaussians.opacities,    # [N]
            colors = gaussians.colors,          # [N, S, 3]
            viewmats = viewmats,                # [B, 4, 4]
            Ks = Ks,                            # [B, 3, 3]
            width=self.image_width, 
            height=self.image_height, 
            sh_degree=gaussians.sh_degree,
            near_plane=self.z_near,
            far_plane=self.z_far,
            tile_size=self.tile_size,
            backgrounds=bg_tensor,              # [B, 3]
            render_mode="RGB+ED", packed=False, distributed=False
        )

        rgb = render_colors[:,:,:,:3]    # [B, H, W, 3]
        depth = render_colors[:,:,:,3:4] # [B, H, W, 1]
        alpha = render_alphas[:,:,:,:1]  # [B, H, W, 1]

        # rgb = composite_with_bg(rgb, alpha, bg_color = self.bg_color)

        return rgb, depth, alpha


