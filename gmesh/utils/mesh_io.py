#
# Peizhi Yan
# Copyright (C) 2026
#

"""
Loading meshes (OBJ) and texture maps, without extra dependencies.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from gmesh.scenes.mesh import Mesh


def load_obj(file_path: str, device="cpu") -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Reads the vertex positions, texture coordinates and faces of an OBJ file
    (polygons are split into triangle fans; normals and materials are ignored).

    Returns:
        verts:    [V, 3] float32
        faces:    [F, 3] int64
        uvs:      [T, 2] float32 texture coordinates, or None if the file has none
        faces_uv: [F, 3] int64 indices into uvs per face corner, or None
    """
    verts, uvs, faces, faces_uv = [], [], [], []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:3]])
            elif parts[0] == "f":
                corners = [c.split("/") for c in parts[1:]]
                v_ids = [_obj_index(c[0], len(verts)) for c in corners]
                t_ids = [_obj_index(c[1], len(uvs)) if len(c) > 1 and c[1] else -1 for c in corners]
                for k in range(1, len(corners) - 1):  # triangle fan
                    faces.append([v_ids[0], v_ids[k], v_ids[k + 1]])
                    faces_uv.append([t_ids[0], t_ids[k], t_ids[k + 1]])

    verts = torch.tensor(verts, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.int64, device=device).view(-1, 3)
    faces_uv = torch.tensor(faces_uv, dtype=torch.int64, device=device).view(-1, 3)
    if len(uvs) == 0 or (faces_uv < 0).any():
        return verts, faces, None, None
    uvs = torch.tensor(uvs, dtype=torch.float32, device=device)
    return verts, faces, uvs, faces_uv


def _obj_index(token: str, count: int) -> int:
    """OBJ indices are 1-based; negative indices count from the end"""
    i = int(token)
    return i - 1 if i > 0 else count + i


def load_texture(file_path: str, device="cpu") -> torch.Tensor:
    """Reads an image as an RGB texture map [Ht, Wt, 3] in [0, 1] (row 0 at the top)."""
    image = np.asarray(Image.open(file_path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).to(device)


def load_textured_mesh(obj_path: str, texture_path: str, device="cpu") -> Mesh:
    """Loads an OBJ file with texture coordinates and its texture map as a textured Mesh."""
    verts, faces, uvs, faces_uv = load_obj(obj_path, device=device)
    assert uvs is not None, f"{obj_path} has no texture coordinates"
    texture = load_texture(texture_path, device=device)
    return Mesh(verts=verts, faces=faces, uvs=uvs, faces_uv=faces_uv, texture=texture)
