#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
from gsplat.rendering import rasterization

from scenes.mesh import Mesh
from scenes.gaussians import Gaussians
from scenes.cameras import PerspectiveCamera
from renderer.renderer_pytorch3d import CustomMeshRenderer
from renderer.rasterizer_gsplat import GaussianRasterizer
from utils.image_utils import blend_rgba


class GmeshRenderer:
    def __init__(self, camera : PerspectiveCamera, tile_size = 16, bg_color = (1.0, 1.0, 1.0)):
        self.camera = camera
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.z_near = camera.z_near
        self.z_far = camera.z_far
        self.tile_size = tile_size      # tile size, as used in 3DGS
        self.bg_color = bg_color        # background color
        self.mesh_renderer = CustomMeshRenderer(camera=camera) # setup mesh renderer
        self.gaussian_rasterizer = GaussianRasterizer(camera=camera) # setup 3DGS rasterizer

    def render(self,
               mesh : Mesh,
               gaussians : Gaussians,
               camera_pose : torch.Tensor,
               return_all : bool = False,
               ):
        """
        Render the scene that contains both 3D Gaussians and 3D Mesh 

        Args:
            mesh (Mesh): Contains vertices, faces, colors
            gaussians (Gaussians): Contains means, quats, scales, opacities, colors, sh_degree.
            camera_pose (torch.Tensor): 
                - for PerspectiveCamera, [B,6] (yaw, pitch, roll, dx, dy, dz);
                - for OrbitCamera, [B, 3] (elevation, azimuth, distance).
            return_all (bool): whether or not to return all rendered results
                
        Returns:
            rgb (torch.Tensor):   [B, H, W, 3]
            depth (torch.Tensor): [B, H, W, 1]
            alpha (torch.Tensor): [B, H, W, 1]

        Notes:
            B is batch size (number of camera views to render), currently only support 1
        """

        # render the 3D mesh
        rgb_m, depth_m, alpha_m = self.mesh_renderer.render(mesh=mesh, camera_pose=camera_pose)

        # render the 3D Gaussians
        rgb_g, depth_g, alpha_g = self.gaussian_rasterizer.rasterize(gaussians=gaussians, camera_pose=camera_pose)

        # blend rendered 3D Gaussians and mesh
        rgb_blend, depth_blend, alpha_blend = blend_rgba(
            rgb_m, depth_m, alpha_m,
            rgb_g, depth_g, alpha_g
        )

        if return_all:
            ret_dict = {
                'mesh':  [rgb_m, depth_m, alpha_m],
                '3dgs':  [rgb_g, depth_g, alpha_g],
                'blend': [rgb_blend, depth_blend, alpha_blend],
            }
            return ret_dict
        else:
            return rgb_blend, depth_blend, alpha_blend


