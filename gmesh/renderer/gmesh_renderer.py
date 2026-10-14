#
# Peizhi Yan
# Copyright (C) 2025
#

import torch

from gmesh.scenes.mesh import Mesh
from gmesh.scenes.gaussians import Gaussians
from gmesh.scenes.cameras import PerspectiveCamera
from gmesh.renderer.renderer_pytorch3d import CustomMeshRenderer
from gmesh.renderer.rasterizer_gsplat import GaussianRasterizer
from gmesh.utils.image_utils import blend_rgba


class GmeshRenderer:
    def __init__(self, camera : PerspectiveCamera, tile_size = 16, bg_color = (1.0, 1.0, 1.0), depth_bias = 0.0):
        self.camera = camera
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.z_near = camera.z_near
        self.z_far = camera.z_far
        self.tile_size = tile_size      # tile size, as used in 3DGS
        self.bg_color = bg_color        # background color
        self.depth_bias = depth_bias    # z-test bias, to avoid z-fighting of Gaussians lying on the mesh surface
        self.mesh_renderer = CustomMeshRenderer(camera=camera, bg_color=bg_color) # setup mesh renderer
        self.gaussian_rasterizer = GaussianRasterizer(camera=camera, tile_size=tile_size, bg_color=bg_color) # setup 3DGS rasterizer

    def render(self,
               mesh : Mesh,
               gaussians : Gaussians,
               camera_pose : torch.Tensor,
               return_all : bool = False,
               mode : str = 'zbuffer',
               depth_bias : float = None,
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
            mode (str):
                - 'zbuffer': each Gaussian is depth-tested against the mesh z-buffer
                  per pixel while alpha compositing; the mesh is the opaque layer
                  behind the visible Gaussians.
                - 'depth_blend': (legacy) render both separately and blend the two
                  images by comparing their depth maps.
            depth_bias (float): z-test bias for 'zbuffer' mode; defaults to self.depth_bias.
                
        Returns:
            rgb (torch.Tensor):   [B, H, W, 3]
            depth (torch.Tensor): [B, H, W, 1]
            alpha (torch.Tensor): [B, H, W, 1]

        Notes:
            B is batch size (number of camera views to render), currently only support 1
        """

        assert mode in ['zbuffer', 'depth_blend'], f"mode should be one of: ['zbuffer', 'depth_blend']"
        if depth_bias is None:
            depth_bias = self.depth_bias

        # render the 3D mesh
        rgb_m, depth_m, alpha_m = self.mesh_renderer.render(mesh=mesh, camera_pose=camera_pose)

        if mode == 'zbuffer' and not return_all:
            rgb_g = depth_g = alpha_g = None
        else:
            # render the 3D Gaussians on their own
            rgb_g, depth_g, alpha_g = self.gaussian_rasterizer.rasterize(gaussians=gaussians, camera_pose=camera_pose)

        if mode == 'zbuffer':
            rgb_blend, depth_blend, alpha_blend = self.composite_with_zbuffer(
                gaussians, camera_pose, rgb_m, depth_m, alpha_m, depth_bias
            )
        else:
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

    def composite_with_zbuffer(self, gaussians, camera_pose, rgb_m, depth_m, alpha_m, depth_bias):
        """
        Differentiable

        Front-to-back alpha compositing of the Gaussians, where each Gaussian is
        depth-tested against the mesh z-buffer at every pixel; the (opaque) mesh,
        or the background where there is no mesh, is the last layer:

            C = sum_{i: z_i < D_m} c_i a_i T_i  +  T_cut * rgb_m

        Args:
            rgb_m:   [B, H, W, 3]  mesh color, already composited over the background
            depth_m: [B, H, W, 1]  mesh camera-space depth
            alpha_m: [B, H, W, 1]  mesh coverage (binary)

        Returns:
            rgb_out:   [B, H, W, 3]
            depth_out: [B, H, W, 1]
            alpha_out: [B, H, W, 1]
        """
        # depth limit for the Gaussians: the mesh surface, or +inf where there is no mesh
        depth_limits = torch.where(alpha_m > 0, depth_m, torch.full_like(depth_m, float('inf')))[..., 0]  # [B, H, W]

        # Gaussians in front of the mesh (premultiplied color, no background)
        rgb_g, depth_acc_g, alpha_g = self.gaussian_rasterizer.rasterize_with_zbuffer(
            gaussians=gaussians, camera_pose=camera_pose,
            depth_limits=depth_limits.detach(), depth_bias=depth_bias,
        )

        # the mesh (or background) is seen through the remaining transmittance
        T = 1.0 - alpha_g
        rgb_out = rgb_g + T * rgb_m
        alpha_out = alpha_g + T * alpha_m

        # expected depth over Gaussians and mesh
        depth_out = (depth_acc_g + T * alpha_m * depth_m) / alpha_out.clamp(min=1e-10)

        return rgb_out, depth_out, alpha_out
