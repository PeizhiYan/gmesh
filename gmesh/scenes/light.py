#
# Peizhi Yan
# Copyright (C) 2026
#

import torch


class PointLight:
    def __init__(self,
                 position = (0.0, 0.0, 10.0),
                 ambient = (0.2, 0.2, 0.4),
                 diffuse = (0.4, 0.4, 0.7),
                 specular = (0.0, 0.0, 0.0),
                 shininess = 30.0,
                 device = "cuda",
                 ):
        """
        A point light for Phong shading of meshes (one light), evaluated per pixel by
        the mesh rasterizer. Values can be tensors (with gradients, to learn the
        lighting) or tuples / numbers.

        At a surface point with base color c (vertex color or texture), normal n,
        light direction l, view direction v (towards the camera) and reflection r of l:

            color = ambient * c + diffuse * c * max(0, n.l) + specular * max(0, r.v) ^ shininess

        clamped to [0, 1].

        Args:
            position:  [3]  light position in world space
            ambient:   [3]  ambient color (0~1.0)
            diffuse:   [3]  diffuse color (0~1.0)
            specular:  [3]  specular color (0~1.0)
            shininess: float, specular exponent
            device:    device of the tensors created from tuples / numbers
        """
        self.position = _to_tensor(position, device)
        self.ambient = _to_tensor(ambient, device)
        self.diffuse = _to_tensor(diffuse, device)
        self.specular = _to_tensor(specular, device)
        self.shininess = _to_tensor(shininess, device)

    def parameters(self):
        """the light's tensors (e.g. to make some of them trainable)"""
        return [self.position, self.ambient, self.diffuse, self.specular, self.shininess]

    def __repr__(self):
        fmt = lambda t: [round(x, 3) for x in t.detach().flatten().tolist()]
        return (f"PointLight(position={fmt(self.position)}, ambient={fmt(self.ambient)}, "
                f"diffuse={fmt(self.diffuse)}, specular={fmt(self.specular)}, shininess={fmt(self.shininess)})")


def _to_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value, dtype=torch.float32, device=device)
