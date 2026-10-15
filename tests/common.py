#
# Peizhi Yan
# Copyright (C) 2025
#

"""Shared helpers for the tests (CUDA GPU required)."""

import math
import os

import pytest
import torch

from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles

from gmesh.renderer.rasterizer_hybrid import camera_space_precisions


DEVICE = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pixel_centers(W, H):
    """[H*W] x and y pixel centers (row-major), as used by the rasterizers."""
    ys, xs = torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing="ij")
    return xs.flatten().float() + 0.5, ys.flatten().float() + 0.5


def random_scene(N=64, W=70, H=45, C=2, tile_size=16, channels=3, seed=0):
    """Random Gaussians seen by C cameras, projected and tile-intersected by gsplat."""
    g = torch.Generator().manual_seed(seed)
    means = (torch.rand(N, 3, generator=g) * 2 - 1) * torch.tensor([1.0, 0.7, 0.5]) + torch.tensor([0.0, 0.0, 3.0])
    quats = torch.randn(N, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, generator=g) * 0.15 + 0.03
    opacities = torch.rand(N, generator=g) * 0.5 + 0.05
    colors = torch.rand(C, N, channels, generator=g)

    viewmats = torch.eye(4).repeat(C, 1, 1)
    viewmats[1:, 0, 3] = 0.2  # shift the other cameras a bit
    Ks = torch.tensor([[60.0, 0.0, W / 2], [0.0, 60.0, H / 2], [0.0, 0.0, 1.0]]).repeat(C, 1, 1)

    to = lambda x: x.to(DEVICE)
    means, quats, scales, opacities, colors, viewmats, Ks = map(to, (means, quats, scales, opacities, colors, viewmats, Ks))

    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales, viewmats, Ks, W, H, packed=False, opacities=opacities
    )
    tile_width = math.ceil(W / tile_size)
    tile_height = math.ceil(H / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, packed=False, n_cameras=C
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # camera-space 3D shape of the Gaussians (for the hybrid rasterizer)
    mus, precs = camera_space_precisions(means, quats, scales, viewmats)

    return dict(
        means2d=means2d.detach(), conics=conics.detach(), colors=colors,
        opacities=opacities.repeat(C, 1), depths=depths.detach(),
        mus=mus.detach(), precs=precs.detach(), Ks=Ks, viewmats=viewmats,
        isect_offsets=isect_offsets, flatten_ids=flatten_ids,
        W=W, H=H, C=C, N=N, tile_size=tile_size,
    )


def leaves(scene, keys=("means2d", "conics", "colors", "opacities")):
    """Fresh leaf tensors (requiring grad) for the differentiable inputs."""
    return [scene[k].clone().requires_grad_(True) for k in keys]


def tile_membership(scene):
    """[C, H*W, N] bool: whether Gaussian n is in the intersection list of the pixel's tile."""
    C, N, W, H, ts = scene["C"], scene["N"], scene["W"], scene["H"], scene["tile_size"]
    offsets = scene["isect_offsets"].flatten().long()
    flatten_ids = scene["flatten_ids"].long()
    n_tiles_total = offsets.numel()
    ends = torch.cat([offsets[1:], offsets.new_tensor([flatten_ids.numel()])])
    tile_of_isect = torch.repeat_interleave(torch.arange(n_tiles_total, device=DEVICE), ends - offsets)
    member = torch.zeros(n_tiles_total, N, dtype=torch.bool, device=DEVICE)
    member[tile_of_isect, flatten_ids % N] = True
    member = member.view(C, -1, N)  # [C, n_tiles, N]

    tile_width = math.ceil(W / ts)
    ys, xs = torch.meshgrid(torch.arange(H, device=DEVICE), torch.arange(W, device=DEVICE), indexing="ij")
    pix_tile = ((ys // ts) * tile_width + (xs // ts)).flatten()  # [H*W]
    return member[:, pix_tile, :]


def assert_close(actual, expected, name, rtol=1e-3, atol_scale=1e-4):
    atol = atol_scale * max(1.0, expected.abs().max().item())
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=lambda m: f"{name}: {m}")


def random_weights(scene, channels, seed=123):
    g = torch.Generator().manual_seed(seed)
    C, H, W = scene["C"], scene["H"], scene["W"]
    w_c = torch.randn(C, H, W, channels, generator=g).to(DEVICE)
    w_a = torch.randn(C, H, W, 1, generator=g).to(DEVICE)
    return w_c, w_a


def load_spot():
    o3d_utils = pytest.importorskip("gmesh.utils.o3d_utils")
    import numpy as np
    vertices, faces = o3d_utils._read_obj_file(os.path.join(ROOT, "assets", "data", "spot.obj"), uv=False)
    theta = np.pi * 0.8
    rotation_matrix = torch.tensor([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ], dtype=torch.float32, device=DEVICE)
    vertices = torch.from_numpy(vertices).float().to(DEVICE) @ rotation_matrix.T
    faces = torch.from_numpy(faces).to(DEVICE)
    return vertices, faces
