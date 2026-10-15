#
# Peizhi Yan
# Copyright (C) 2025
#

"""
Differentiable triangle-mesh rasterizer.

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


def _screen_barycentrics(tri_uv: Tensor, p: Tensor) -> Tensor:
    """Screen-space barycentric coordinates of points p [M, 2] w.r.t. triangles tri_uv [M, 3, 2]."""
    w = torch.stack([
        _edge(tri_uv[:, 1], tri_uv[:, 2], p),
        _edge(tri_uv[:, 2], tri_uv[:, 0], p),
        _edge(tri_uv[:, 0], tri_uv[:, 1], p),
    ], dim=-1)                                               # [M, 3]
    return w / w.sum(dim=-1, keepdim=True)


def face_adjacency(faces: Tensor) -> Tensor:
    """
    [F, 3] index of the face sharing the edge opposite to vertex k of each face
    (-1 for boundary edges). Cached on the faces tensor.
    """
    cached = getattr(faces, "_gmesh_adjacency", None)
    if cached is not None and cached[0] == faces._version and cached[1].device == faces.device:
        return cached[1]
    with torch.no_grad():
        f = faces.long()
        F_ = f.shape[0]
        # edge k is opposite to vertex k: (v1, v2), (v2, v0), (v0, v1)
        edges = torch.stack([f[:, [1, 2]], f[:, [2, 0]], f[:, [0, 1]]], dim=1).sort(dim=-1).values  # [F, 3, 2]
        n_verts = int(f.max().item()) + 1 if F_ > 0 else 1
        keys = (edges[..., 0] * n_verts + edges[..., 1]).flatten()  # [3F]
        sorted_keys, order = torch.sort(keys)
        same = sorted_keys[1:] == sorted_keys[:-1]
        i0, i1 = order[:-1][same], order[1:][same]
        adjacency = torch.full((3 * F_,), -1, dtype=torch.long, device=faces.device)
        adjacency[i0] = i1 // 3
        adjacency[i1] = i0 // 3
        adjacency = adjacency.view(F_, 3)
    try:
        faces._gmesh_adjacency = (faces._version, adjacency)
    except Exception:
        pass
    return adjacency


def _antialias(
    face_ids: Tensor,   # [C, H, W]
    depth: Tensor,      # [C, H, W, 1]
    color: Optional[Tensor],  # [C, H, W, A] or None
    uv: Tensor,         # [C, V, 2]
    verts_z: Tensor,    # [C, V] camera-space depth of the vertices
    faces: Tensor,      # [F, 3]
    adjacency: Tensor,  # [F, 3]
):
    """
    Analytic edge antialiasing (in the spirit of nvdiffrast's antialias op).

    For every pair of horizontally / vertically adjacent pixels that see a
    geometric discontinuity (different triangles that are not neighbors across
    the crossed edge, or a triangle next to an empty pixel), the silhouette edge
    of the front triangle T crosses the segment between the two pixel centers
    at a fraction t (from the pixel covered by T). With a 1D box filter:
      - t < 0.5: the pixel covered by T loses (0.5 - t) of its coverage, to the
        surface behind it or to the background;
      - t > 0.5: the other pixel gains (t - 0.5) of T.
    t depends on the projected vertices, so this gives fractional coverage at
    silhouettes and gradients w.r.t. the silhouette vertex positions.

    Returns premultiplied color, coverage alpha and fragment depth.
    """
    C, H, W = face_ids.shape
    device = face_ids.device
    n_pix = C * H * W
    ids = face_ids.reshape(-1).long()
    covered = ids >= 0
    depth_flat = depth.reshape(-1)
    pix = torch.arange(n_pix, device=device).view(C, H, W)

    # adjacent pixel pairs (a, b): horizontal and vertical
    pa = torch.cat([pix[:, :, :-1].reshape(-1), pix[:, :-1, :].reshape(-1)])
    pb = torch.cat([pix[:, :, 1:].reshape(-1), pix[:, 1:, :].reshape(-1)])
    horizontal = torch.cat([
        torch.ones(C * H * (W - 1), dtype=torch.bool, device=device),
        torch.zeros(C * (H - 1) * W, dtype=torch.bool, device=device),
    ])
    keep = (ids[pa] != ids[pb]) & (covered[pa] | covered[pb])
    pa, pb, horizontal = pa[keep], pb[keep], horizontal[keep]

    # front pixel f (covered by the nearer triangle T) and other pixel o
    a_front = ~covered[pb] | (covered[pa] & (depth_flat[pa] <= depth_flat[pb]))
    pf = torch.where(a_front, pa, pb)
    po = torch.where(a_front, pb, pa)
    T = ids[pf]
    o_covered = covered[po]

    cam = pf // (H * W)
    def centers(q):
        rem = q % (H * W)
        return torch.stack([rem % W, rem // W], dim=-1).to(uv.dtype) + 0.5

    with torch.no_grad():
        b_o = _screen_barycentrics(uv[cam[:, None], faces[T]], centers(po))  # [P, 3]
        b_f = _screen_barycentrics(uv[cam[:, None], faces[T]], centers(pf))
        # the segment leaves T through the edge with the smallest crossing t
        t_all = b_f / (b_f - b_o)
        t_all = torch.where(b_o < 0, t_all, torch.full_like(t_all, float("inf")))
        k = t_all.argmin(dim=-1)                                   # [P]
        exits = torch.isfinite(t_all.gather(1, k[:, None])[:, 0])
        # continuous surface between the two pixels: no discontinuity. The other
        # pixel's triangle U is the neighbor across the crossed edge, shares a
        # vertex with T, or lies on T's plane (within 1% depth)
        U = ids[po].clamp(min=0)
        shares_vertex = (faces[T][:, :, None] == faces[U][:, None, :]).flatten(1).any(dim=1)
        tri_iz = 1.0 / verts_z[cam[:, None], faces[T]]                       # [P, 3]
        z_T_at_o = 1.0 / (b_o * tri_iz).sum(dim=-1).clamp(min=1e-12)
        same_plane = (z_T_at_o - depth_flat[po]).abs() <= 1e-2 * depth_flat[po]
        continuous = o_covered & ((adjacency[T, k] == ids[po]) | shares_vertex | same_plane)
        discontinuity = exits & ~continuous
        # each edge is handled along one axis only (as in nvdiffrast): horizontal
        # pairs for edges closer to vertical, vertical pairs for the others
        tri_k = uv[cam[:, None], faces[T]]                                 # [P, 3, 2]
        e0 = tri_k.gather(1, ((k + 1) % 3)[:, None, None].expand(-1, 1, 2))[:, 0]
        e1 = tri_k.gather(1, ((k + 2) % 3)[:, None, None].expand(-1, 1, 2))[:, 0]
        e_dir = (e1 - e0).abs()
        discontinuity = discontinuity & torch.where(horizontal, e_dir[:, 0] <= e_dir[:, 1], e_dir[:, 0] > e_dir[:, 1])
    sel = discontinuity.nonzero(as_tuple=True)[0]
    pf, po, T, k, cam, o_covered = pf[sel], po[sel], T[sel], k[sel], cam[sel], o_covered[sel]

    # differentiable crossing fraction t along the edge k of T
    tri_uv = uv[cam[:, None], faces[T]]                            # [P, 3, 2]
    b_f = _screen_barycentrics(tri_uv, centers(pf)).gather(1, k[:, None])[:, 0]
    b_o = _screen_barycentrics(tri_uv, centers(po)).gather(1, k[:, None])[:, 0]
    t = (b_f / (b_f - b_o)).clamp(0.0, 1.0)
    gain = (t - 0.5).clamp(min=0.0)  # coverage of T spilling into pixel o
    loss = (0.5 - t).clamp(min=0.0)  # coverage of pixel f not covered by T

    alpha = covered.to(depth.dtype)
    # f -> background: lose coverage; o (empty) gains coverage of T
    alpha = alpha.index_add(0, pf, -(loss * (~o_covered).to(loss.dtype)))
    alpha = alpha.index_add(0, po, gain * (~o_covered).to(gain.dtype))

    # depth of the newly covered (previously empty) pixels: from the front pixel
    empty_w = gain * (~o_covered).to(gain.dtype)
    w_sum = torch.zeros_like(alpha).index_add(0, po, empty_w)
    d_sum = torch.zeros_like(alpha).index_add(0, po, empty_w * depth_flat[pf])
    depth_out = torch.where(covered, depth_flat, d_sum / w_sum.clamp(min=1e-12))

    alpha_raw = alpha
    alpha = alpha_raw.clamp(0.0, 1.0)
    depth_out = torch.where(alpha > 0, depth_out, torch.zeros_like(depth_out))

    premult = None
    if color is not None:
        A = color.shape[-1]
        col = color.reshape(-1, A)
        P = col * covered[:, None].to(col.dtype)
        oc = o_covered[:, None].to(col.dtype)
        # f: part of the pixel shows the background (lost) or the surface behind (mixed)
        P = P.index_add(0, pf, loss[:, None] * (oc * (col[po] - col[pf]) - (1 - oc) * col[pf]))
        # o: part of the pixel shows T (gained over background, or mixed over the surface behind)
        P = P.index_add(0, po, gain[:, None] * (col[pf] - oc * col[po]))
        # renormalize where several pairs over/under-filled the pixel
        scale = torch.where(alpha_raw > 1, 1.0 / alpha_raw.clamp(min=1e-12),
                            torch.where(alpha_raw < 0, torch.zeros_like(alpha_raw), torch.ones_like(alpha_raw)))
        premult = (P * scale[:, None]).view(C, H, W, A)

    return premult, alpha.view(C, H, W, 1), depth_out.view(C, H, W, 1)


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
    antialias: bool = False,
) -> Dict[str, Tensor]:
    """
    Rasterize a triangle mesh from C cameras.

    Returns a dict with:
        face_ids:   [C, H, W]    int32, nearest face per pixel, -1 where empty
        mask:       [C, H, W, 1] float, 1 where a face covers the pixel center
        bary:       [C, H, W, 3] perspective-correct barycentric coordinates (differentiable)
        attributes: [C, H, W, A] interpolated attributes at the pixel center, 0 where empty
        alpha:      [C, H, W, 1] mesh coverage: fractional at silhouettes if antialias, else = mask
        depth:      [C, H, W, 1] camera-space depth of the mesh fragment, 0 where alpha == 0
        color:      [C, H, W, A] attributes premultiplied by the coverage (antialiased if antialias)
    All outputs except face_ids and mask are differentiable w.r.t. verts and attributes.
    """
    assert verts.dim() == 2 and verts.shape[1] == 3, verts.shape
    assert faces.dim() == 2 and faces.shape[1] == 3, faces.shape
    C = viewmats.shape[0]
    faces_long = faces.long()

    verts_cam, uv = project_vertices(verts, viewmats, Ks)  # [C, V, 3], [C, V, 2]

    # 1) visibility (not differentiable)
    with torch.no_grad():
        face_z = verts_cam[..., 2][:, faces_long]           # [C, F, 3]
        face_uv = uv[:, faces_long]                          # [C, F, 3, 2]
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
    tri = faces_long[face_ids[cam_idx, ii, jj].long()]      # [M, 3]
    p = torch.stack([jj, ii], dim=-1).to(uv.dtype) + 0.5    # [M, 2]
    tri_uv = uv[cam_idx[:, None], tri]                       # [M, 3, 2]
    tri_z = verts_cam[cam_idx[:, None], tri, 2]              # [M, 3]

    bary_screen = _screen_barycentrics(tri_uv, p)
    q = bary_screen / tri_z                                  # 1/z is affine in screen space
    inv_depth = q.sum(dim=-1, keepdim=True)
    depth_m = 1.0 / inv_depth                                # [M, 1]
    bary_m = q * depth_m                                     # perspective-correct

    flat = (cam_idx * height + ii) * width + jj              # [M]
    n_pix = C * height * width

    def scatter(values: Tensor) -> Tensor:
        out = values.new_zeros(n_pix, values.shape[-1])
        return out.index_put((flat,), values).view(C, height, width, values.shape[-1])

    mask = covered[..., None].to(verts.dtype)
    depth = scatter(depth_m)
    out = {
        "face_ids": face_ids,
        "mask": mask,
        "bary": scatter(bary_m),
    }
    color = None
    if attributes is not None:
        assert attributes.dim() == 2 and attributes.shape[0] == verts.shape[0], attributes.shape
        attr_m = (bary_m[..., None] * attributes[tri]).sum(dim=1)  # [M, A]
        color = scatter(attr_m)
        out["attributes"] = color

    if antialias:
        premult, alpha, depth_aa = _antialias(
            face_ids, depth, color, uv, verts_cam[..., 2], faces_long, face_adjacency(faces)
        )
        out["alpha"] = alpha
        out["depth"] = depth_aa
        if premult is not None:
            out["color"] = premult
    else:
        out["alpha"] = mask
        out["depth"] = depth
        if color is not None:
            out["color"] = color  # already 0 where empty
    return out


class MeshRenderer:
    """
    Renders a Mesh with per-vertex colors.
    """

    def __init__(self, camera : PerspectiveCamera, tile_size = 16, bg_color = (1.0, 1.0, 1.0), antialias = True):
        self.camera = camera
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.z_near = camera.z_near
        self.z_far = camera.z_far
        self.tile_size = tile_size
        self.bg_color = bg_color        # background color
        self.antialias = antialias      # analytic edge antialiasing (fractional coverage at silhouettes)

    def rasterize(self, mesh : Mesh, camera_pose : torch.Tensor) -> Dict[str, Tensor]:
        """
        Returns the raw rasterization (see `rasterize_mesh`), with the vertex
        colors as attributes ('color': [B, H, W, 3] premultiplied by the coverage 'alpha').
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
            antialias=self.antialias,
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
            alpha (torch.Tensor): [B, H, W, 1]  mesh coverage (fractional at silhouettes if antialiased)
        """
        raster = self.rasterize(mesh, camera_pose)
        alpha = raster["alpha"]
        rgb = raster["color"] + (1.0 - alpha) * _bg_tensor(self.bg_color, alpha)
        return rgb, raster["depth"], alpha


def _bg_tensor(bg_color, like: Tensor) -> Tensor:
    if not isinstance(bg_color, torch.Tensor):
        bg_color = torch.tensor(bg_color, dtype=like.dtype, device=like.device)
    return bg_color.to(dtype=like.dtype, device=like.device).view(1, 1, 1, -1)
