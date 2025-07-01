#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
from scenes.mesh import Mesh
from scenes.gaussians import Gaussians


def mesh_to_gaussians(mesh : Mesh, method='vertex'):
    """
    We currently support three methods to convert the triangle mesh
    to a 3D Gaussians representation:
        1) 'vertex': convert each vertex to a 3D Gaussian
        2) 'facet': assian one 3D Gaussian to the center of each triangle facet
        3) 'facet3': assian three 3D Gaussians to each triangle facet
    """
    assert method in ['vertex', 'facet', 'facet3'], f"Method should be one of: [\'vertex\', \'facet\', \'facet3\']"

    if method == 'vertex':
        return _mesh_to_gaussians_vertex(mesh)
    elif method == 'facet':
        return _mesh_to_gaussians_facet(mesh)
    elif method == 'facet3':
        return _mesh_to_gaussians_facet3(mesh)



def _mesh_to_gaussians_vertex(mesh : Mesh):
    device = mesh.device

    means = torch.clone(mesh.verts) # [V, 3]    
    num_gaussians = len(means)
    sh_degree = 0
    quats = torch.zeros([num_gaussians, 4], dtype=torch.float32).to(device)
    quats[:,0] = 1.0
    scales = 0.02 * torch.ones([num_gaussians, 3], dtype=torch.float32).to(device)
    opacities = 1.0 * torch.ones([num_gaussians], dtype=torch.float32).to(device)
    colors = 0.5 * torch.ones([num_gaussians, (sh_degree + 1)**2, 3], dtype=torch.float32).to(device)

    # create the gaussians object
    gaussians = Gaussians(means=means,
                        quats=quats,
                        scales=scales,
                        opacities=opacities,
                        colors=colors)
    return gaussians


def _mesh_to_gaussians_facet(mesh : Mesh):
    pass


def _mesh_to_gaussians_facet3(mesh : Mesh):
    pass


