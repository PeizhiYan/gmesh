#
# Peizhi Yan
# Copyright (C) 2025
#

import torch

class Mesh:
    def __init__(self,
                 verts : torch.tensor,
                 faces : torch.tensor,
                 colors : torch.tensor = None,
                 uvs : torch.tensor = None,
                 faces_uv : torch.tensor = None,
                 texture : torch.tensor = None,
                 allow_check : bool = True,
                 ):
        """
        All inputs should be torch tensors (with gradients if needed).

        The mesh color comes from one of two sources:
            - vertex colors (`colors`), interpolated over the faces, or
            - a texture map (`texture`, with texture coordinates `uvs`), used
              whenever a texture is given.

        Args:
            verts:     [V, 3]    torch.Tensor
            faces:     [F, 3]    torch.Tensor
            colors:    [V, 3]    torch.Tensor    per-vertex RGB color (range 0~1.0)
            uvs:       [T, 2]    torch.Tensor    texture coordinates in [0, 1] (OBJ convention: v = 0 at the bottom of the image)
            faces_uv:  [F, 3]    torch.Tensor    indices into uvs per face corner (as the "vt" indices of OBJ faces);
                                                 defaults to faces (one uv per vertex, T = V)
            texture:   [Ht, Wt, 3] torch.Tensor  RGB texture map (range 0~1.0), row 0 at the top of the image.
                                                 Used as given: a tensor with requires_grad (or a torch.nn.Parameter)
                                                 is trainable, like vertex colors and positions
            allow_check: whether to validate the input tensors. Boolean

        Notes:
            V is the number of vertices
            F is the number of triangle faces
            T is the number of texture coordinates
        """

        # meta data
        self.num_vertices = verts.shape[0]
        self.num_faces = faces.shape[0]

        if texture is not None and faces_uv is None:
            faces_uv = faces

        if allow_check:
            # validate tensor shapes
            assert verts.dim() == 2 and verts.shape[1] == 3, \
                f"verts should have shape (V, 3), but got {verts.shape}"
            assert faces.dim() == 2 and faces.shape[1] == 3, \
                f"faces should have shape (F, 3), but got {faces.shape}"
            if colors is not None:
                assert colors.shape == verts.shape, \
                    f"colors should have shape {verts.shape}, but got {colors.shape}"
            if texture is not None:
                assert texture.dim() == 3 and texture.shape[2] == 3, \
                    f"texture should have shape (Ht, Wt, 3), but got {texture.shape}"
                assert uvs is not None, "a textured mesh needs texture coordinates (uvs)"
                assert uvs.dim() == 2 and uvs.shape[1] == 2, \
                    f"uvs should have shape (T, 2), but got {uvs.shape}"
                assert faces_uv.shape == faces.shape, \
                    f"faces_uv should have shape {faces.shape}, but got {faces_uv.shape}"

        # 3D mesh data
        self.device = verts.device
        self.verts = verts
        self.faces = faces
        if colors is None:
            self.colors = 0.5 * torch.ones_like(self.verts).detach().to(self.device)
        else:
            self.colors = colors

        # texture map (optional)
        self.uvs = uvs
        self.faces_uv = faces_uv
        self.texture = texture

    @property
    def has_texture(self) -> bool:
        """True if the mesh is colored by its texture map (otherwise by its vertex colors)"""
        return self.texture is not None

    def __repr__(self):
        color = f"Texture={tuple(self.texture.shape)}" if self.has_texture else "VertexColors"
        return (f"Mesh(Vertices={self.num_vertices}, Faces={self.num_faces}, {color})")
