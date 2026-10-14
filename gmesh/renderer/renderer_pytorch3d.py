#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
from pytorch3d.renderer import (
    OrthographicCameras, MeshRenderer, MeshRasterizer,
    SoftPhongShader, RasterizationSettings, PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from gmesh.utils.graphics_utils import *
from gmesh.utils.image_utils import composite_with_bg
from gmesh.scenes.mesh import Mesh
from gmesh.scenes.cameras import PerspectiveCamera


class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments

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
            image_size=(self.image_height, self.image_width),
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

        # PyTorch3D's NDC convention for non-square images: the shorter side
        # spans [-1, 1] and the longer side spans [-s, s] (s = aspect ratio).
        # Rendering at the native (H, W) keeps the mesh pixel-aligned with gsplat.
        min_side = min(self.image_height, self.image_width)
        ndc_scale = torch.tensor([self.image_width / min_side, self.image_height / min_side, 1.0],
                                 dtype=verts_ndc_3d.dtype, device=verts_ndc_3d.device)
        verts_ndc_3d = verts_ndc_3d * ndc_scale

        # remap NDC z from [-1, 1] to [0, 1]: PyTorch3D discards fragments with
        # z < 0, which would cull geometry near the camera. The map is affine,
        # so the screen-space interpolated z-buffer stays exact.
        verts_ndc_3d = torch.cat([verts_ndc_3d[..., :2], 0.5 * (verts_ndc_3d[..., 2:3] + 1.0)], dim=-1)

        # create Pytorch3D mesh object for rendering
        pytorch3d_mesh = Meshes(verts=verts_ndc_3d, faces=faces)
        pytorch3d_mesh.textures = TexturesVertex(verts_features=colors)

        # render image with depth map
        image, fragments = mesh_renderer(pytorch3d_mesh)  # [N, H, W, C+1]

        # pixels covered by the mesh; we currently do not support mesh opacity
        alpha = (fragments.pix_to_face[..., :1] >= 0).to(torch.float32).detach()  # [N, H, W, 1]

        # convert depth from NDC to real camera-space depth, 0 where there is no mesh
        depth = ndc_depth_to_real_depth(2.0 * fragments.zbuf[..., :1] - 1.0, znear=self.z_near, zfar=self.z_far)
        depth = depth * alpha  # [N, H, W, 1]

        # RGB image
        rgb = image[..., :3]

        # # mask out empty pixels
        # rgb = rgb * alpha

        # set background color
        rgb = composite_with_bg(rgb, alpha, bg_color = self.bg_color)

        return rgb, depth, alpha


