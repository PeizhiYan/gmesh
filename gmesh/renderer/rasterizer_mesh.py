#
# Peizhi Yan
# Copyright (C) 2025
#

"""
Differentiable triangle-mesh rasterizer (no PyTorch3D).

1. Vertices are projected with the same pinhole camera (viewmat + K) and pixel
   convention (pixel centers at +0.5) as gsplat.
2. Triangles are binned into image tiles (PyTorch) and a CUDA kernel picks the
   nearest triangle at each pixel center (z-buffer). This visibility decision
   is the only non-differentiable step.
3. Barycentric coordinates, depth and vertex attributes are recomputed in
   PyTorch from the face ids, with perspective-correct interpolation, so
   autograd gives gradients w.r.t. vertex positions and vertex attributes.

Triangles with a vertex in front of the near plane are skipped (no clipping).
"""

import math
from typing import Dict, Optional

import torch
from torch import Tensor

from gmesh.renderer._backend import load_extension
from gmesh.scenes.mesh import Mesh
from gmesh.scenes.cameras import PerspectiveCamera
from gmesh.utils.image_utils import composite_with_bg


def project_vertices(verts: Tensor, viewmats: Tensor, Ks: Tensor):
    """
    Args:
        verts:    [V, 3] world-space vertices
        viewmats: [C, 4, 4] world-to-camera (OpenCV: x right, y down, z forward)
        Ks:       [C, 3, 3]

    Returns:
        verts_cam: [C, V, 3] camera-space vertices
        uv:        [C, V, 2] pixel coordinates (pixel centers at +0.5, as in gsplat)
    """
    R = viewmats[:, :3, :3]  # [C, 3, 3]
    t = viewmats[:, :3, 3]   # [C, 3]
    verts_cam = torch.einsum("cij,vj->cvi", R, verts) + t[:, None, :]
    z = verts_cam[..., 2:3]
    # avoid 0 / 0 in the backward pass for (unused) vertices on the camera plane
    z = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
    focal = torch.stack([Ks[:, 0, 0], Ks[:, 1, 1]], dim=-1)[:, None, :]      # [C, 1, 2]
    principal = torch.stack([Ks[:, 0, 2], Ks[:, 1, 2]], dim=-1)[:, None, :]  # [C, 1, 2]
    uv = verts_cam[..., :2] / z * focal + principal
    return verts_cam, uv


@torch.no_grad()
def _bin_faces(face_uv: Tensor, face_valid: Tensor, width: int, height: int, tile_size: int):
    """
    Assign triangles to the image tiles their pixel-center bounding box overlaps.

    Args:
        face_uv:    [C, F, 3, 2] projected triangle corners
        face_valid: [C, F] triangles to rasterize

    Returns:
        tile_offsets: [C, tile_height, tile_width] int32, start of each tile's range
        flatten_ids:  [n_isects] int32, indices into [C * F], sorted by tile
    """
    device = face_uv.device
    C, F = face_valid.shape
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    n_tiles = tile_width * tile_height

    uv = torch.where(face_valid[..., None, None], face_uv, torch.zeros_like(face_uv))
    umin, umax = uv[..., 0].amin(-1), uv[..., 0].amax(-1)
    vmin, vmax = uv[..., 1].amin(-1), uv[..., 1].amax(-1)

    # range of pixels whose centers (j + 0.5) can be inside the triangle
    jmin = torch.ceil(umin - 0.5).clamp(-1, width)
    jmax = torch.floor(umax - 0.5).clamp(-1, width)
    imin = torch.ceil(vmin - 0.5).clamp(-1, height)
    imax = torch.floor(vmax - 0.5).clamp(-1, height)
    valid = (
        face_valid
        & torch.isfinite(face_uv).all(dim=-1).all(dim=-1)
        & (jmax >= jmin) & (imax >= imin)
        & (jmax >= 0) & (jmin <= width - 1)
        & (imax >= 0) & (imin <= height - 1)
    )
    jmin = jmin.clamp(0, width - 1).long()
    jmax = jmax.clamp(0, width - 1).long()
    imin = imin.clamp(0, height - 1).long()
    imax = imax.clamp(0, height - 1).long()

    tx0, tx1 = jmin // tile_size, jmax // tile_size
    ty0, ty1 = imin // tile_size, imax // tile_size
    nx = (tx1 - tx0 + 1).flatten()
    ny = (ty1 - ty0 + 1).flatten()
    counts = torch.where(valid.flatten(), nx * ny, torch.zeros_like(nx))  # [C * F]

    total = int(counts.sum().item())
    if total == 0:
        tile_offsets = torch.zeros(C, tile_height, tile_width, dtype=torch.int32, device=device)
        flatten_ids = torch.zeros(0, dtype=torch.int32, device=device)
        return tile_offsets, flatten_ids

    face_flat = torch.repeat_interleave(torch.arange(C * F, device=device), counts)  # [total]
    starts = torch.cumsum(counts, dim=0) - counts
    local = torch.arange(total, device=device) - starts[face_flat]
    nx_f = nx[face_flat]
    tx = tx0.flatten()[face_flat] + local % nx_f
    ty = ty0.flatten()[face_flat] + local // nx_f
    cam = face_flat // F
    tile_ids = cam * n_tiles + ty * tile_width + tx

    order = torch.argsort(tile_ids, stable=True)
    flatten_ids = face_flat[order].to(torch.int32).contiguous()

    tile_counts = torch.bincount(tile_ids, minlength=C * n_tiles)
    tile_offsets = (torch.cumsum(tile_counts, dim=0) - tile_counts).to(torch.int32)
    tile_offsets = tile_offsets.view(C, tile_height, tile_width).contiguous()
    return tile_offsets, flatten_ids


def _edge(a: Tensor, b: Tensor, p: Tensor) -> Tensor:
    """Twice the signed area of (a, b, p). a, b, p: [..., 2]"""
    return (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (p[..., 0] - a[..., 0])


def rasterize_mesh(
    verts: Tensor,       # [V, 3]
    faces: Tensor,       # [F, 3]
    viewmats: Tensor,    # [C, 4, 4]
    Ks: Tensor,          # [C, 3, 3]
    width: int,
    height: int,
    attributes: Optional[Tensor] = None,  # [V, A] per-vertex attributes (e.g. colors)
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    tile_size: int = 16,
) -> Dict[str, Tensor]:
    """
    Rasterize a triangle mesh from C cameras.

    Returns a dict with:
        face_ids:   [C, H, W]    int32, nearest face per pixel, -1 where empty
        mask:       [C, H, W, 1] float, 1 where a face covers the pixel
        depth:      [C, H, W, 1] camera-space depth (z), 0 where empty (differentiable)
        bary:       [C, H, W, 3] perspective-correct barycentric coordinates (differentiable)
        attributes: [C, H, W, A] interpolated attributes, 0 where empty (differentiable)
    """
    assert verts.dim() == 2 and verts.shape[1] == 3, verts.shape
    assert faces.dim() == 2 and faces.shape[1] == 3, faces.shape
    device = verts.device
    C = viewmats.shape[0]
    faces = faces.long()

    verts_cam, uv = project_vertices(verts, viewmats, Ks)  # [C, V, 3], [C, V, 2]

    # 1) visibility (not differentiable)
    with torch.no_grad():
        face_z = verts_cam[..., 2][:, faces]                # [C, F, 3]
        face_uv = uv[:, faces]                               # [C, F, 3, 2]
        face_valid = (face_z > near_plane).all(dim=-1) & (face_z < far_plane).any(dim=-1)
        face_data = torch.cat(
            [face_uv.reshape(C, -1, 6), 1.0 / face_z.clamp(min=near_plane)], dim=-1
        ).float().contiguous()                               # [C, F, 9]
        tile_offsets, flatten_ids = _bin_faces(face_uv, face_valid, width, height, tile_size)
        face_ids, _ = load_extension().rasterize_mesh_fwd(
            face_data, float(far_plane), width, height, tile_size, tile_offsets, flatten_ids
        )                                                    # [C, H, W]

    # 2) barycentrics, depth and attributes (differentiable)
    covered = face_ids >= 0
    cam_idx, ii, jj = covered.nonzero(as_tuple=True)        # [M]
    tri = faces[face_ids[cam_idx, ii, jj].long()]           # [M, 3]
    p = torch.stack([jj, ii], dim=-1).to(uv.dtype) + 0.5    # [M, 2]
    tri_uv = uv[cam_idx[:, None], tri]                       # [M, 3, 2]
    tri_z = verts_cam[cam_idx[:, None], tri, 2]              # [M, 3]

    w = torch.stack([
        _edge(tri_uv[:, 1], tri_uv[:, 2], p),
        _edge(tri_uv[:, 2], tri_uv[:, 0], p),
        _edge(tri_uv[:, 0], tri_uv[:, 1], p),
    ], dim=-1)                                               # [M, 3]
    bary_screen = w / w.sum(dim=-1, keepdim=True)
    q = bary_screen / tri_z                                  # 1/z is affine in screen space
    inv_depth = q.sum(dim=-1, keepdim=True)
    depth_m = 1.0 / inv_depth                                # [M, 1]
    bary_m = q * depth_m                                     # perspective-correct

    flat = (cam_idx * height + ii) * width + jj              # [M]
    n_pix = C * height * width

    def scatter(values: Tensor) -> Tensor:
        out = values.new_zeros(n_pix, values.shape[-1])
        return out.index_put((flat,), values).view(C, height, width, values.shape[-1])

    out = {
        "face_ids": face_ids,
        "mask": covered[..., None].to(verts.dtype),
        "depth": scatter(depth_m),
        "bary": scatter(bary_m),
    }
    if attributes is not None:
        assert attributes.dim() == 2 and attributes.shape[0] == verts.shape[0], attributes.shape
        attr_m = (bary_m[..., None] * attributes[tri]).sum(dim=1)  # [M, A]
        out["attributes"] = scatter(attr_m)
    return out


class MeshRenderer:
    """
    Renders a Mesh with per-vertex colors (drop-in replacement of the former
    PyTorch3D-based CustomMeshRenderer).
    """

    def __init__(self, camera : PerspectiveCamera, tile_size = 16, bg_color = (1.0, 1.0, 1.0)):
        self.camera = camera
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.z_near = camera.z_near
        self.z_far = camera.z_far
        self.tile_size = tile_size
        self.bg_color = bg_color        # background color

    def rasterize(self, mesh : Mesh, camera_pose : torch.Tensor) -> Dict[str, Tensor]:
        """
        Returns the raw rasterization (see `rasterize_mesh`), with the vertex
        colors as attributes ('attributes': [B, H, W, 3], not composited).
        """
        batch_size = camera_pose.shape[0]
        Ks = self.camera.get_intrinsic_matrix(batch_size=batch_size)  # [B, 3, 3]
        viewmats = self.camera.get_view_matrix(camera_pose)            # [B, 4, 4]
        return rasterize_mesh(
            verts=mesh.verts,
            faces=mesh.faces,
            viewmats=viewmats,
            Ks=Ks,
            width=self.image_width,
            height=self.image_height,
            attributes=mesh.colors,
            near_plane=self.z_near,
            far_plane=self.z_far,
            tile_size=self.tile_size,
        )

    def render(self, mesh : Mesh, camera_pose : torch.Tensor):
        """
        Render the mesh from multiple camera views.

        Args:
            mesh (Mesh): Mesh object.
            camera_pose (torch.Tensor):
                - for PerspectiveCamera, [B,6] (yaw, pitch, roll, dx, dy, dz);
                - for OrbitCamera, [B, 3] (elevation, azimuth, distance).

        Returns:
            rgb (torch.Tensor):   [B, H, W, 3]  composited over the background color
            depth (torch.Tensor): [B, H, W, 1]  camera-space depth, 0 where there is no mesh
            alpha (torch.Tensor): [B, H, W, 1]  mesh coverage
        """
        raster = self.rasterize(mesh, camera_pose)
        alpha = raster["mask"]
        rgb = composite_with_bg(raster["attributes"], alpha, bg_color=self.bg_color)
        return rgb, raster["depth"], alpha
