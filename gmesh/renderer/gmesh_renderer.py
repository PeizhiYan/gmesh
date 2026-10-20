#
# Peizhi Yan
# Copyright (C) 2026
#

import torch

from gmesh.scenes.mesh import Mesh
from gmesh.scenes.gaussians import Gaussians
from gmesh.scenes.cameras import PerspectiveCamera
from gmesh.renderer.rasterizer_mesh import MeshRenderer
from gmesh.renderer.rasterizer_gaussians import GaussianRasterizer
from gmesh.renderer.rasterizer_hybrid import composite_hybrid


class GmeshRenderer:
    def __init__(self, 
                 camera : PerspectiveCamera, 
                 tile_size = 16, 
                 bg_color = (1.0, 1.0, 1.0), 
                 softness = 1.0,
                 antialias = True, 
                 cull_occluded = True):
        """
        Differentiable hybrid renderer for 3D Gaussians and triangle meshes.

        Args:
            camera (PerspectiveCamera): camera (intrinsics, image size, near/far planes).
            tile_size (int): tile size of the rasterizers, as in 3DGS.
            bg_color (tuple): background color.
            softness (float): width of the soft depth test between Gaussians and the mesh,
                relative to each Gaussian's own depth extent (smaller is sharper).
            antialias (bool): analytic antialiasing of the mesh edges (fractional coverage
                and gradients at silhouettes).
            cull_occluded (bool): skip Gaussians certainly hidden behind tiles fully covered
                by the mesh (does not change the result, saves time).
        """
        self.camera = camera
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.z_near = camera.z_near
        self.z_far = camera.z_far
        self.tile_size = tile_size          # tile size, as used in 3DGS
        self.bg_color = bg_color            # background color
        self.softness = softness            # width of the soft depth test, relative to the Gaussians' extent
        self.cull_occluded = cull_occluded  # skip Gaussians certainly hidden behind fully covered tiles
        self.mesh_renderer = MeshRenderer(camera=camera, tile_size=tile_size, bg_color=bg_color,
                                          antialias=antialias) # setup mesh renderer (antialiased mesh edges)
        self.gaussian_rasterizer = GaussianRasterizer(camera=camera, tile_size=tile_size, bg_color=bg_color) # setup 3DGS rasterizer

    def render(self,
               mesh : Mesh,
               gaussians : Gaussians,
               camera_pose : torch.Tensor,
               return_all : bool = False,
               softness : float = None,
               ):
        """
        Render the scene that contains both 3D Gaussians and 3D Mesh

        At every pixel, each Gaussian is split into the part in front of and the
        part behind the mesh surface (soft depth test using the Gaussian's 3D
        shape); the two parts are composited in front of / behind the opaque,
        antialiased mesh. This is smooth where Gaussians cross the surface and at
        mesh silhouettes, and differentiable w.r.t. the Gaussians and the mesh.

        Args:
            mesh (Mesh): Contains vertices, faces, colors
            gaussians (Gaussians): Contains means, quats, scales, opacities, colors, sh_degree.
            camera_pose (torch.Tensor):
                - for PerspectiveCamera, [B,6] (yaw, pitch, roll, dx, dy, dz);
                - for OrbitCamera, [B, 3] (elevation, azimuth, distance).
            return_all (bool): whether or not to also return the mesh and the 3D Gaussians rendered on their own
            softness (float): soft depth test width; defaults to self.softness.

        Returns:
            rgb (torch.Tensor):   [B, H, W, 3]
            depth (torch.Tensor): [B, H, W, 1]
            alpha (torch.Tensor): [B, H, W, 1]
            or, with return_all, a dict {'mesh': [rgb, depth, alpha], '3dgs': [...], 'blend': [...]}

        Notes:
            B is batch size (number of camera views to render), currently only support 1
        """
        if softness is None:
            softness = self.softness

        # render the 3D mesh
        mesh_raster = self.mesh_renderer.rasterize(mesh=mesh, camera_pose=camera_pose)
        alpha_m = mesh_raster['alpha']        # [B, H, W, 1] coverage (fractional at silhouettes if antialiased)
        depth_m = mesh_raster['depth']        # [B, H, W, 1]
        color_m = mesh_raster['color']        # [B, H, W, 3] premultiplied by the coverage

        # render the 3D Gaussians in front of / behind the mesh surface
        depth_limits = torch.where(alpha_m > 0, depth_m, torch.full_like(depth_m, float('inf')))[..., 0]
        layers = self.gaussian_rasterizer.rasterize_hybrid(
            gaussians=gaussians, camera_pose=camera_pose,
            depth_limits=depth_limits, coverages=alpha_m[..., 0], softness=softness,
            cull_occluded=self.cull_occluded,
        )

        # composite: Gaussians in front, the mesh, then the Gaussians behind it and the background
        rgb, depth, alpha = composite_hybrid(layers, color_m, depth_m, alpha_m, self.bg_color)

        if return_all:
            bg = torch.tensor(self.bg_color, dtype=color_m.dtype, device=color_m.device).view(1, 1, 1, 3)
            rgb_m = color_m + (1.0 - alpha_m) * bg  # mesh alone, composited over the background
            rgb_g, depth_g, alpha_g = self.gaussian_rasterizer.rasterize(gaussians=gaussians, camera_pose=camera_pose)
            ret_dict = {
                'mesh':  [rgb_m, depth_m, alpha_m],
                '3dgs':  [rgb_g, depth_g, alpha_g],
                'blend': [rgb, depth, alpha],
            }
            return ret_dict
        else:
            return rgb, depth, alpha
