#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
from gsplat.rendering import rasterization

from gmesh.scenes.gaussians import Gaussians
from gmesh.scenes.cameras import PerspectiveCamera
# from utils.graphics_utils import build_view_matrix
from gmesh.renderer.rasterizer_hybrid import rasterization_hybrid

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

        return rgb, depth, alpha

    def rasterize_hybrid(self,
                         gaussians : Gaussians,
                         camera_pose : torch.Tensor,
                         depth_limits : torch.Tensor,
                         coverages : torch.Tensor,
                         softness : float = 1.0,
                         cull_occluded : bool = True,
                         ):
        """
        Rasterize 3D Gaussians into two layers, in front of and behind a mesh
        surface, with a soft per-ray depth test (see rasterizer_hybrid.py).

        Args:
            gaussians (Gaussians): Contains means, quats, scales, opacities, colors, sh_degree.
            camera_pose (torch.Tensor):
                - for PerspectiveCamera, [B,6] (yaw, pitch, roll, dx, dy, dz);
                - for OrbitCamera, [B, 3] (elevation, azimuth, distance).
            depth_limits (torch.Tensor): [B, H, W] mesh camera-space depth; +inf where there is no mesh.
            coverages (torch.Tensor): [B, H, W] mesh coverage in [0, 1].
            softness (float): scales the width of the soft depth test (1.0 = the Gaussian's own extent).
            cull_occluded (bool): skip Gaussians certainly hidden behind fully covered tiles (exact, faster).

        Returns:
            dict with 'front', 'back' ([B, H, W, 4]: RGB + accumulated depth) and
            'T_front', 'T_back' ([B, H, W, 1] transmittances).
        """
        batch_size = camera_pose.shape[0]
        Ks = self.camera.get_intrinsic_matrix(batch_size=batch_size) # [B, 3, 3]
        viewmats = self.camera.get_view_matrix(camera_pose)          # [B, 4, 4]

        return rasterization_hybrid(
            means = gaussians.means,            # [N, 3]
            quats = gaussians.quats,            # [N, 4]
            scales = gaussians.scales,          # [N, 3]
            opacities = gaussians.opacities,    # [N]
            colors = gaussians.colors,          # [N, S, 3]
            sh_degree = gaussians.sh_degree,
            viewmats = viewmats,                # [B, 4, 4]
            Ks = Ks,                            # [B, 3, 3]
            width = self.image_width,
            height = self.image_height,
            depth_limits = depth_limits,        # [B, H, W]
            coverages = coverages,              # [B, H, W]
            softness = softness,
            cull_occluded = cull_occluded,
            near_plane = self.z_near,
            far_plane = self.z_far,
            tile_size = self.tile_size,
        )
