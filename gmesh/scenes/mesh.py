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
                 allow_check : bool = True,
                 ):
        """
        All inputs should be torch tensors (with gradients if needed).

        Args:
            verts:     [V, 3]    torch.Tensor
            faces:     [F, 3]    torch.Tensor
            colors:    [V, 3]    torch.Tensor    per-vertex RGB color (range 0~1.0)
            allow_check: whether to validate the input tensors. Boolean

        Notes:
            V is the number of vertices
            F is the number of triangle faces 
        """

        # meta data
        self.num_vertices = verts.shape[0]
        self.num_faces = faces.shape[0]

        if allow_check:
            # validate tensor shapes
            assert verts.dim() == 2 and verts.shape[1] == 3, \
                f"verts should have shape (V, 3), but got {verts.shape}"
            assert faces.dim() == 2 and faces.shape[1] == 3, \
                f"faces should have shape (F, 3), but got {faces.shape}"
            if colors is not None:
                assert colors.shape == verts.shape, \
                    f"colors should have shape {verts.shape}, but got {colors.shape}"

        # 3D mesh data
        self.device = verts.device
        self.verts = verts
        self.faces = faces
        if colors is None:
            self.colors = 0.5 * torch.ones_like(self.verts).detach().to(self.device)
        else:
            self.colors = colors

    def __repr__(self):
        return (f"Mesh(Vertices={self.num_vertices}, Faces={self.num_faces})")

