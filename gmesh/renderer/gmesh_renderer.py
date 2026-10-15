#
# Peizhi Yan
# Copyright (C) 2025
#

import torch

from gmesh.scenes.mesh import Mesh
from gmesh.scenes.gaussians import Gaussians
from gmesh.scenes.cameras import PerspectiveCamera
from gmesh.renderer.rasterizer_mesh import MeshRenderer
from gmesh.renderer.rasterizer_gsplat import GaussianRasterizer
from gmesh.renderer.rasterizer_hybrid import composite_hybrid
from gmesh.utils.image_utils import blend_rgba


class GmeshRenderer:
    def __init__(self, camera : PerspectiveCamera, tile_size = 16, bg_color = (1.0, 1.0, 1.0), depth_bias = 0.0, softness = 1.0,
                 antialias = True, cull_occluded = True):
        self.camera = camera
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.z_near = camera.z_near
        self.z_far = camera.z_far
        self.tile_size = tile_size      # tile size, as used in 3DGS
        self.bg_color = bg_color        # background color
        self.depth_bias = depth_bias    # 'zbuffer' mode: z-test bias, to avoid z-fighting of Gaussians lying on the mesh surface
        self.softness = softness        # 'hybrid' mode: width of the soft depth test, relative to the Gaussians' extent
        self.cull_occluded = cull_occluded  # 'hybrid' mode: skip Gaussians certainly hidden behind fully covered tiles
        self.mesh_renderer = MeshRenderer(camera=camera, tile_size=tile_size, bg_color=bg_color,
                                          antialias=antialias) # setup mesh renderer (antialiased mesh edges)
        self.gaussian_rasterizer = GaussianRasterizer(camera=camera, tile_size=tile_size, bg_color=bg_color) # setup 3DGS rasterizer

    def render(self,
               mesh : Mesh,
               gaussians : Gaussians,
               camera_pose : torch.Tensor,
               return_all : bool = False,
               mode : str = 'hybrid',
               depth_bias : float = None,
               softness : float = None,
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
                - 'hybrid': at every pixel, each Gaussian is split into the part in front of
                  and the part behind the mesh surface (soft depth test using the Gaussian's
                  3D shape); the two parts are composited in front of / behind the opaque
                  (antialiased) mesh. Smooth where Gaussians cross the surface and at mesh
                  silhouettes, and differentiable w.r.t. Gaussian and mesh geometry.
                - 'zbuffer': each Gaussian is depth-tested by its center depth against
                  the mesh z-buffer per pixel (hard test).
                - 'depth_blend': (legacy) render both separately and blend the two
                  images by comparing their depth maps.
            depth_bias (float): z-test bias for 'zbuffer' mode; defaults to self.depth_bias.
            softness (float): soft depth test width for 'hybrid' mode; defaults to self.softness.
                
        Returns:
            rgb (torch.Tensor):   [B, H, W, 3]
            depth (torch.Tensor): [B, H, W, 1]
            alpha (torch.Tensor): [B, H, W, 1]

        Notes:
            B is batch size (number of camera views to render), currently only support 1
        """

        assert mode in ['hybrid', 'zbuffer', 'depth_blend'], f"mode should be one of: ['hybrid', 'zbuffer', 'depth_blend']"
        if depth_bias is None:
            depth_bias = self.depth_bias
        if softness is None:
            softness = self.softness

        # render the 3D mesh
        mesh_raster = self.mesh_renderer.rasterize(mesh=mesh, camera_pose=camera_pose)
        alpha_m = mesh_raster['alpha']        # [B, H, W, 1] coverage (fractional at silhouettes if antialiased)
        depth_m = mesh_raster['depth']        # [B, H, W, 1]
        color_m = mesh_raster['color']        # [B, H, W, 3] premultiplied by the coverage
        bg = torch.tensor(self.bg_color, dtype=color_m.dtype, device=color_m.device).view(1, 1, 1, 3)
        rgb_m = color_m + (1.0 - alpha_m) * bg  # [B, H, W, 3] composited over the background

        if mode in ['hybrid', 'zbuffer'] and not return_all:
            rgb_g = depth_g = alpha_g = None
        else:
            # render the 3D Gaussians on their own
            rgb_g, depth_g, alpha_g = self.gaussian_rasterizer.rasterize(gaussians=gaussians, camera_pose=camera_pose)

        if mode == 'hybrid':
            # depth of the mesh surface, +inf where there is no mesh
            depth_limits = torch.where(alpha_m > 0, depth_m, torch.full_like(depth_m, float('inf')))[..., 0]
            layers = self.gaussian_rasterizer.rasterize_hybrid(
                gaussians=gaussians, camera_pose=camera_pose,
                depth_limits=depth_limits, coverages=alpha_m[..., 0], softness=softness,
                cull_occluded=self.cull_occluded,
            )
            rgb_blend, depth_blend, alpha_blend = composite_hybrid(
                layers, color_m, depth_m, alpha_m, self.bg_color
            )
        elif mode == 'zbuffer':
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
            alpha_m: [B, H, W, 1]  mesh coverage

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
