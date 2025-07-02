#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
import torch.nn.functional as F
from pytorch3d.renderer import (
    OrthographicCameras, MeshRenderer, MeshRasterizer,
    SoftPhongShader, RasterizationSettings, PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from utils.graphics_utils import *
from utils.image_utils import composite_with_bg
from scenes.mesh import Mesh
from scenes.cameras import PerspectiveCamera


class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf

def resize_back(image, size):
    """
    Resize input image of shape [N, H', W', C] to [N, H, W, C]

    Note:   iterpolation mode `nearest` will reduce the boundary artifacts 
            (inaccurate depth value) in the depth map. 
    """
    image = image.permute(0, 3, 1, 2) # [N, C, H, H]
    image = F.interpolate(image, size=size, mode='nearest') # [N, C, H, W]
    image = image.permute(0, 2, 3, 1)
    return image

def ndc_depth_to_real_depth(depth, znear, zfar):
    """
    Converts depth buffer (in NDC [-1, 1]) to real camera-space Z.
    """
    real_depth = 2.0 * znear * zfar / (zfar + znear - depth * (zfar - znear))
    return real_depth


class CustomMeshRenderer:
    def __init__(self, camera : PerspectiveCamera, bg_color = (1.0,1.0,1.0)):
        self.camera = camera
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.z_near = camera.z_near
        self.z_far = camera.z_far
        self.bg_color = bg_color        # background color

        self.raster_settings = RasterizationSettings(
            image_size=self.image_height,
            blur_radius=0.0,
            faces_per_pixel=1
        )

    def get_batch_dummy_cameras(self, batch_size, device):
        # no camera transform: 
        # this is a hack; set camera at the origin with identity rotation
        R = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device) # [B, 3, 3]
        R[:, 0, 0] = -1.0
        R[:, 1, 1] = -1.0
        T = torch.zeros(batch_size, 3, device=device) # [B, 3]
        pytorch3d_dummy_cameras = OrthographicCameras(device=device, R=R, T=T)
        return pytorch3d_dummy_cameras
    
    def get_batch_dummy_lights(self, batch_size, device):
        # no real light: only apply ambient light
        # user needs to compute the real vertex color for shading by themself
        location = torch.tensor([[0.0, 0.0, 0.0]], device=device).repeat(batch_size, 1)
        ambient_color = torch.tensor([[1.0, 1.0, 1.0]], device=device).repeat(batch_size, 1)
        diffuse_color = torch.tensor([[0.0, 0.0, 0.0]], device=device).repeat(batch_size, 1)
        specular_color = torch.tensor([[0.0, 0.0, 0.0]], device=device).repeat(batch_size, 1)
        pytorch_3d_dummy_lights = PointLights(
            device=device, 
            location=location,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color
        )
        return pytorch_3d_dummy_lights

    def render(self,
               mesh : Mesh,
               camera_pose : torch.Tensor,
               ):
        """
        Rasterize 3D Gaussians from multiple camera views.

        Args:
            mesh (Mesh): Mesh object.
            camera_pose (torch.Tensor): 
                - for PerspectiveCamera, [B,6] (yaw, pitch, roll, dx, dy, dz);
                - for OrbitCamera, [B, 3] (elevation, azimuth, distance).

        Returns:
            rgb (torch.Tensor):   [B, H, W, 3]
            depth (torch.Tensor): [B, H, W, 1]
            alpha (torch.Tensor): [B, H, W, 1]

        Notes:
            B is batch size (number of camera views to render)
            But currently we do not support batch rendering (ensure B is 1 for now)!!
        """
        batch_size = camera_pose.shape[0]
        device = camera_pose.device
        
        verts = mesh.verts[None]   # extend to [B, V, 3], B is 1
        faces = mesh.faces[None]   # extend to [B, F, 3], B is 1
        colors = mesh.colors[None] # extend to [B, V, 3], B is 1

        # create mesh rasterizer
        pytorch3d_dummy_cameras = self.get_batch_dummy_cameras(batch_size, device)
        mesh_rasterizer = MeshRasterizer(cameras=pytorch3d_dummy_cameras, raster_settings=self.raster_settings)

        # create mesh renderer
        pytorch3d_dummy_lights = self.get_batch_dummy_lights(batch_size, device)
        mesh_renderer = MeshRendererWithDepth(
            rasterizer=mesh_rasterizer,
            shader=SoftPhongShader(device=device, cameras=pytorch3d_dummy_cameras, lights=pytorch3d_dummy_lights)
        )

        # get intrinsic matrix
        Ks = self.camera.get_intrinsic_matrix(batch_size=batch_size) # [B, 3, 3]

        # build view matrix
        viewmats = self.camera.get_view_matrix(camera_pose) # [B, 4, 4]

        # project the vertices to 2D
        verts_clip = batch_perspective_projection_with_view_matrix(verts, viewmats, Ks, 
                                                                   self.image_height, self.image_width, 
                                                                   self.z_near, self.z_far)
        verts_ndc_3d = batch_verts_clip_to_ndc(verts_clip)  # output [B, V, 3] normalized to -1.0 ~ 1.0

        # create Pytorch3D mesh object for rendering
        pytorch3d_mesh = Meshes(verts=verts_ndc_3d, faces=faces)
        pytorch3d_mesh.textures = TexturesVertex(verts_features=colors)

        # render image with depth map
        image, depth = mesh_renderer(pytorch3d_mesh)      # [N, H, H, C+1], [N, H, H, 1]

        # convert depth from NDC to realworld
        depth = ndc_depth_to_real_depth(depth, znear=self.z_near, zfar=self.z_far)
        
        # trick: for non-square image, we first render the iamge in square, 
        # then resize it to match the expected size
        image = resize_back(image, size=(self.image_height, self.image_width)) # [N, C, H, W]
        depth = resize_back(depth, size=(self.image_height, self.image_width)) # [N, C, H, W]

        # separate RGB image and alpha map    
        rgb = image[..., :3]
        alpha = image[..., 3:4]

        # we currently do not support mesh opacity
        alpha = (alpha > 0).to(torch.float32).detach()

        # # mask out empty pixels
        # rgb = rgb * alpha

        # set background color
        rgb = composite_with_bg(rgb, alpha, bg_color = self.bg_color)

        return rgb, depth, alpha


