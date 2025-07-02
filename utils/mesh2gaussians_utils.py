#
# Peizhi Yan
# Copyright (C) 2025
#

import torch
from scenes.mesh import Mesh
from scenes.gaussians import Gaussians


def mesh_to_gaussians(mesh : Mesh, method='vertex', trainable : bool = False):
    """
    We currently support three methods to convert the triangle mesh to 3D Gaussians:
        1) 'vertex': convert each vertex to a 3D Gaussian
        2) 'facet': assian one 3D Gaussian to the center of each triangle facet
        3) 'facet3': assian three 3D Gaussians to each triangle facet

    Args:
        mesh:       Mesh object
        method:     str     indicates method to convert mesh into 3D Gaussians
        trainable:  bool    whether or not to let the Gaussian attributes be trainable
    
    Returns:
        gaussians:  Gaussians object
    """
    assert method in ['vertex', 'facet', 'facet3'], f"Method should be one of: [\'vertex\', \'facet\', \'facet3\']"

    if method == 'vertex':
        means, quats, scales, opacities, colors = _mesh_to_gaussians_vertex(mesh)
    elif method == 'facet':
        means, quats, scales, opacities, colors = _mesh_to_gaussians_facet(mesh)
    elif method == 'facet3':
        means, quats, scales, opacities, colors = _mesh_to_gaussians_facet3(mesh)

    if trainable:
        # convert torch.Tensor to torch.nn.Parameter 
        means = torch.nn.Parameter(means)
        quats = torch.nn.Parameter(quats)
        scales = torch.nn.Parameter(scales)
        opacities = torch.nn.Parameter(opacities)
        colors = torch.nn.Parameter(colors)

    # create the gaussians object
    gaussians = Gaussians(means=means,
                          quats=quats,
                          scales=scales,
                          opacities=opacities,
                          colors=colors)
    return gaussians


def _mesh_to_gaussians_vertex(mesh : Mesh):
    # takes mesh, and returns Gaussians attributes
    device = mesh.device
    means = torch.clone(mesh.verts) # [V, 3]    
    num_gaussians = len(means)
    sh_degree = 0
    quats = torch.zeros([num_gaussians, 4], dtype=torch.float32).to(device)
    quats[:,0] = 1.0
    scales = 0.02 * torch.ones([num_gaussians, 3], dtype=torch.float32).to(device)
    opacities = 1.0 * torch.ones([num_gaussians], dtype=torch.float32).to(device)
    colors = 0.5 * torch.ones([num_gaussians, (sh_degree + 1)**2, 3], dtype=torch.float32).to(device)
    return means, quats, scales, opacities, colors



def _mesh_to_gaussians_facet(mesh : Mesh):
    pass


def _mesh_to_gaussians_facet3(mesh : Mesh):
    pass


