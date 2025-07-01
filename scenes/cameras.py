#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
import numpy as np

from utils.graphics_utils import *


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
        viewmats = torch.linalg.inv(Rt)     # [B, 4, 4]
        return viewmats


class OrbitCamera(PerspectiveCamera):
    def __init__(self):
        super().__init__()
        pass

