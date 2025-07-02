#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
from utils.graphics_utils import compute_vertex_normals


def simple_phone_shading(vertices : torch.Tensor, 
                         faces : torch.Tensor, 
                         base_color : torch.Tensor = None,
                         light_pos : torch.Tensor = None,
                         camera_pos : torch.Tensor = None,
                         ambient_color : torch.Tensor = None,
                         diffuse_color : torch.Tensor = None,
                         specular_color : torch.Tensor = None,
                         shininess : int = 30,
                         ):
    """
    Assume there is only one light

    Args:
        - vertices:         [V, 3]
        - faces:            [F, 3]
        - base_color:       [V, 3]  0~1.0
        - light_pos:        [3]
        - camera_pos:       [3]
        - ambient_color:    [3]     0~1.0
        - diffuse_color:    [3]     0~1.0
        - specular_color:   [3]     0~1.0
        - shininess:        int

    Returns:
        - shaded_color:     [V, 3]  0~1.0
    """

    normals = compute_vertex_normals(verts=vertices, faces=faces)           # [V, 3]

    # defaults
    if light_pos is None:
        light_pos = torch.tensor([0.0, 0.0, 10.0], device=vertices.device)      # In front of the mesh
    if camera_pos is None:
        camera_pos = torch.tensor([0.0, 0.0, 10.0], device=vertices.device)
    if ambient_color is None:
        ambient_color  = torch.tensor([0.2, 0.2, 0.4], device=vertices.device)  # [3], RGB
    if diffuse_color is None:
        diffuse_color  = torch.tensor([0.4, 0.4, 0.7], device=vertices.device)
    if specular_color is None:
        specular_color = torch.tensor([0.0, 0.0, 0.0], device=vertices.device)
    if base_color is None:
        base_color = torch.ones_like(vertices) 
    
    # Get vertices for a single mesh
    v = vertices     # [V, 3]
    n = normals      # [V, 3]

    # Light direction (from vertex to light)
    light_dir = (light_pos - v)  # [V, 3]
    light_dir = light_dir / (light_dir.norm(dim=-1, keepdim=True) + 1e-8)

    # View direction (from vertex to camera)
    view_dir = (camera_pos - v)
    view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)

    # Reflection direction for specular
    reflect_dir = 2 * (n * (n * light_dir).sum(-1, keepdim=True)) - light_dir
    reflect_dir = reflect_dir / (reflect_dir.norm(dim=-1, keepdim=True) + 1e-8)

    # Ambient term: just base color * ambient
    ambient = ambient_color * base_color  # [V, 3]

    # Diffuse term: base color * diffuse * max(0, dot(normal, light_dir))
    diff = torch.clamp((n * light_dir).sum(-1, keepdim=True), min=0.0)
    diffuse = diffuse_color * base_color * diff  # [V, 3]

    # Specular term: specular color * [max(0, dot(reflect_dir, view_dir))^shininess]
    spec = torch.clamp((reflect_dir * view_dir).sum(-1, keepdim=True), min=0.0)
    specular = specular_color * (spec ** shininess)  # [V, 3]

    # Final color per vertex
    shaded_color = ambient + diffuse + specular
    shaded_color = torch.clamp(shaded_color, 0.0, 1.0)  # Ensure in [0,1]

    return shaded_color

