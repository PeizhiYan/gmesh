#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
import numpy as np

from gmesh.utils.graphics_utils import *

def look_at_matrix(eye, target, up):
    # eye, target, up: [B, 3]
    z_axis = torch.nn.functional.normalize(eye - target, dim=-1)  # [B, 3]
    x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis), dim=-1)
    y_axis = torch.cross(z_axis, x_axis)

    # [B, 3, 3]
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    t = -torch.bmm(R, eye.unsqueeze(-1)).squeeze(-1)  # [B, 3]
    Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # [B, 3, 4]
    # Homogeneous
    bottom = torch.tensor([0, 0, 0, 1], dtype=Rt.dtype, device=Rt.device).expand(Rt.shape[0], 1, 4)
    Rt = torch.cat([Rt, bottom], dim=1)  # [B, 4, 4]
    return Rt

def invert_rt_batch(Rt: torch.Tensor) -> torch.Tensor:
    """
    Analytic inverse for batched rigid transforms [R|t; 0 0 0 1].
    Rt: [B, 4, 4]
    """
    R = Rt[:, :3, :3]                      # [B, 3, 3]
    t = Rt[:, :3, 3:4]                     # [B, 3, 1]
    R_inv = R.transpose(-1, -2).contiguous()

    top = torch.cat([R_inv, t_inv], dim=-1)  # [B, 3, 4]
    bottom = torch.zeros(Rt.shape[0], 1, 4, dtype=Rt.dtype, device=Rt.device)
    bottom[..., 0, 3] = 1
    return torch.cat([top, bottom], dim=1)   # [B, 4, 4]



class PerspectiveCamera:
    def __init__(self, image_width=512, image_height=512, fov_x=20, fov_y=20, z_near=0.1, z_far=100, device='cuda'):
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.image_width = image_width
        self.image_height = image_height
        self.focal_x = fov_to_focal(fov = self.fov_x, sensor_size = self.image_width)
        self.focal_y = fov_to_focal(fov = self.fov_y, sensor_size = self.image_height)
        self.z_near = z_near
        self.z_far = z_far
        self.device = device

    def get_intrinsic_matrix(self, batch_size=1):
        Ks = build_intrinsics(batch_size=batch_size, image_height=self.image_height, image_width=self.image_width, 
                              focal_x=self.focal_x, focal_y=self.focal_y, device=self.device) # Ks is [B, 3, 3]
        return Ks

    def get_view_matrix(self, camera_pose : torch.Tensor):
        if camera_pose.ndim != 2 or camera_pose.shape[1] != 6:
            print("Error: camera_pose must have shape [B, 6], where B is the batch size, the values are yaw, pitch, roll, dx, dy, dz")
        Rt = build_view_matrix(camera_pose) # [B, 4, 4]
        Rt[:, :, [1,2]] *= -1               # OpenCV: flip y, z
        # viewmats = torch.linalg.inv(Rt)   # [B, 4, 4]
        viewmats = invert_rt_batch(Rt)      # [B, 4, 4]
        return viewmats


class OrbitCamera(PerspectiveCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_camera_offsets(self, yaw, pitch, radius):
        dx = torch.sin(yaw) * radius * 0.9
        dy = -torch.sin(pitch) * radius * 0.9
        dz_a = radius - torch.cos(yaw) * radius
        dz_b = radius - torch.cos(pitch) * radius
        dz = dz_a + dz_b
        return dx, dy, dz

    def get_view_matrix(self, camera_pose: torch.Tensor):
        """
        camera_pose: [B, 3], (elevation, azimuth, distance)
        - elevation: angle from Y axis (radians)
        - azimuth: angle around Y axis (radians)
        - distance: distance from origin
        Returns:
            [B, 4, 4] view matrix (world to camera)
        """
        B = camera_pose.shape[0]
        elevation = camera_pose[:, 0]  # [B]
        azimuth   = camera_pose[:, 1]  # [B]
        distance  = camera_pose[:, 2]  # [B]

        yaw = azimuth
        pitch = elevation
        roll = torch.zeros_like(yaw)

        dx, dy, dz = self.compute_camera_offsets(yaw, pitch, distance)
        # torch.stack to shape [B, 6]
        camera_6dof = torch.stack([yaw, pitch, roll, dx, dy, distance - dz], dim=-1)

        Rt = build_view_matrix(camera_6dof)  # [B, 4, 4]
        Rt[:, :, [1,2]] *= -1                # OpenCV/graphics convention fix

        # viewmats = torch.linalg.inv(Rt)    # [B, 4, 4]
        viewmats = invert_rt_batch(Rt)       # [B, 4, 4]
        return viewmats
