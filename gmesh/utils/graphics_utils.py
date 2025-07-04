#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
import math
import numpy as np
import cv2


def compute_vertex_normals(verts, faces):
    # verts: [V, 3]
    # faces: [F, 3]
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)  # [F, 3]
    face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + 1e-8)
    # Accumulate per-vertex normals
    normals = torch.zeros_like(verts)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    return normals


def euler_to_matrix(yaw, pitch, roll):
    """Convert yaw, pitch, roll to a rotation matrix (B, 3, 3)"""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    zeros = torch.zeros_like(cy)
    ones = torch.ones_like(cy)
    Rz = torch.stack([
        torch.stack([cr, -sr, zeros], dim=-1),
        torch.stack([sr, cr, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)
    Ry = torch.stack([
        torch.stack([cy, zeros, sy], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sy, zeros, cy], dim=-1)
    ], dim=-2)
    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cp, -sp], dim=-1),
        torch.stack([zeros, sp, cp], dim=-1)
    ], dim=-2)
    return Rz @ Ry @ Rx


def build_view_matrix(camera_pose):
    """Build [B, 4, 4] world-to-view matrix from [B, 6] pose"""
    B = camera_pose.shape[0]
    yaw, pitch, roll = camera_pose[:, 0], camera_pose[:, 1], camera_pose[:, 2]
    tx, ty, tz = camera_pose[:, 3], camera_pose[:, 4], camera_pose[:, 5]
    R = euler_to_matrix(yaw, pitch, roll)
    #T = torch.stack([tx, ty, tz], dim=-1).unsqueeze(-1)  # [B, 3, 1]
    Rt = torch.eye(4, device=camera_pose.device, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
    Rt[:, :3, :3] = R
    Rt[:, :3, 3] = torch.stack([tx, ty, tz], dim=-1)
    return Rt  # [B, 4, 4]


def fov_to_focal(fov, sensor_size):
    fov_rad = math.radians(fov)
    focal_length = 0.5 * sensor_size / math.tan(0.5 * fov_rad)
    return focal_length


def build_intrinsics(batch_size, image_height, image_width, focal_x, focal_y, device='cuda'):
    H = image_height
    W = image_width
    fx = focal_x
    fy = focal_y
    cx = W / 2.0
    cy = H / 2.0
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ], dtype=torch.float32, device=device)
    K = K.unsqueeze(0).expand(batch_size, -1, -1).clone()
    return K


def projection_from_intrinsics(K, image_height, image_width, z_near=0.1, z_far=10.0, z_sign=1):
    B = K.shape[0]
    h = image_height
    w = image_width
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    proj = torch.zeros((B, 4, 4), device=K.device, dtype=K.dtype)
    proj[:, 0, 0] = 2 * fx / w
    proj[:, 1, 1] = 2 * fy / h
    proj[:, 0, 2] = (w - 2 * cx) / w
    proj[:, 1, 2] = (h - 2 * cy) / h
    proj[:, 2, 2] = z_sign * (z_far + z_near) / (z_far - z_near)
    proj[:, 2, 3] = -2 * z_far * z_near / (z_far - z_near)
    proj[:, 3, 2] = z_sign
    return proj


def batch_perspective_projection_with_view_matrix(verts, viewmats, K, image_height, image_width, near=0.1, far=10.0):
    """
        - verts: [B, V, 3]
        - viewmats: [B, 4, 4]
        - K: [B, 3, 3] or [1, 3, 3]
    """
    B = verts.shape[0]
    device = verts.device

    world2view = viewmats

    # Build projection matrix
    proj = projection_from_intrinsics(K, image_height, image_width, near, far, z_sign=1) # [B, 4, 4] or [1, 4, 4]

    if proj.shape[0] == 1 and world2view.shape[0] > 1:
        # Expand proj (when proj is [1, 4, 4] but batch size > 1) to match batch size
        proj = proj.expand(world2view.shape[0], -1, -1)
    
    # Full transform (proj @ view)
    full_proj = torch.bmm(proj, world2view)      # [B, 4, 4]

    # Flip y after projection
    full_proj[:, 1, :] *= -1

    # Project verts
    verts_h = torch.cat([verts, torch.ones(B, verts.shape[1], 1, device=device)], dim=-1)  # [B, V, 4]
    verts_clip = torch.bmm(verts_h, full_proj.transpose(-1, -2)) # [B, V, 4]
    return verts_clip


def batch_perspective_projection(verts, camera_pose, K, image_height, image_width, near=0.1, far=10.0):
    """
        - verts: [B, V, 3]
        - camera_pose: [B, 6]
        - K: [B, 3, 3] or [1, 3, 3]
    """

    # build world-to-view matrix
    Rt = build_view_matrix(camera_pose)   # [B, 4, 4]
    Rt[:, :, [1,2]] *= -1                 # OpenCV: flip y, z
    viewmats = torch.linalg.inv(Rt)       # [B, 4, 4]

    verts_clip = batch_perspective_projection_with_view_matrix(verts, viewmats, K, image_height, image_width, near, far)
    return verts_clip


def batch_verts_clip_to_ndc(verts_clip):
    ndc = verts_clip[:, :, :3] / (verts_clip[:, :, 3:4] + 1e-8)
    ndc[:, :, 1] *= -1  # flip y
    return ndc  # [B, V, 3]


def batch_verts_ndc_to_screen(ndc, image_height, image_width):
    screen = (ndc / 2.0 + 0.5) 
    screen[:,:,0] *= image_width
    screen[:,:,1] *= image_height
    return screen




