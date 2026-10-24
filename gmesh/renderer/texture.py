#
# Peizhi Yan
# Copyright (C) 2026
#

"""
Differentiable texture sampling for the mesh rasterizer.

Texture coordinates (u, v) are in [0, 1] with the OBJ convention (v = 0 at the
bottom of the image); the texture is an image [Ht, Wt, C] with row 0 at the top.

Filters:
    "bilinear":  bilinear interpolation of the full-resolution texture.
    "mipmap":    trilinear filtering (bilinear in the two nearest levels of a
                 mipmap pyramid, blended linearly), with the level of detail of
                 each pixel from the screen-space derivatives of its texture
                 coordinates, as on GPUs. Avoids aliasing when the texture is
                 minified and spreads the gradients over all the texels a pixel
                 covers (with "bilinear", a minified texture only gets gradients
                 at 4 texels per pixel).

Both are differentiable w.r.t. the texture and the texture coordinates (and so
the vertex positions); the level of detail itself is not differentiated.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


TEXTURE_FILTERS = ("bilinear", "mipmap")


def build_mipmaps(texture: Tensor) -> List[Tensor]:
    """
    Mipmap pyramid of a texture [Ht, Wt, C] (differentiable).

    Returns a list of [1, C, h, w] levels, from the full resolution down to
    1x1, each the 2x2 box-filtered average of the previous one.
    """
    level = texture.permute(2, 0, 1)[None]  # [1, C, Ht, Wt]
    levels = [level]
    while level.shape[-1] > 1 or level.shape[-2] > 1:
        level = F.avg_pool2d(level, kernel_size=2, stride=2, ceil_mode=True)
        levels.append(level)
    return levels


def _bilinear(level: Tensor, texcoords: Tensor) -> Tensor:
    """level [1, C, h, w], texcoords [M, 2] -> [M, C]"""
    # u -> x in [-1, 1] (left to right), v -> y in [-1, 1] (v = 1 at the top row)
    grid = torch.stack([texcoords[:, 0] * 2.0 - 1.0, 1.0 - texcoords[:, 1] * 2.0], dim=-1)
    out = F.grid_sample(level, grid.view(1, 1, -1, 2).to(level.dtype), mode="bilinear",
                        padding_mode="border", align_corners=False)  # [1, C, 1, M]
    return out[0, :, 0].t()


def sample_texture(
    texture: Tensor,             # [Ht, Wt, C]
    texcoords: Tensor,           # [M, 2]
    lod: Optional[Tensor] = None,  # [M] level of detail (for "mipmap")
    filter: str = "mipmap",
) -> Tensor:
    """
    Samples the texture at the texture coordinates. Returns [M, C].

    With filter "mipmap", `lod` is the mipmap level of each sample (0 is the
    full resolution, 1 half of it, ...; see `texture_lod`).
    """
    assert filter in TEXTURE_FILTERS, f"filter should be one of {TEXTURE_FILTERS}, got {filter}"
    if texcoords.shape[0] == 0:
        return texture.new_zeros(0, texture.shape[-1])
    if filter == "bilinear" or lod is None:
        return _bilinear(texture.permute(2, 0, 1)[None], texcoords)

    levels = build_mipmaps(texture)
    n = len(levels)
    lod = lod.detach().clamp(0.0, n - 1.0)
    l0 = lod.floor().long().clamp(max=n - 1)
    l1 = (l0 + 1).clamp(max=n - 1)
    frac = (lod - l0.to(lod.dtype))[:, None].to(texture.dtype)

    out = texture.new_zeros(texcoords.shape[0], texture.shape[-1])
    for l in range(n):
        lo = (l0 == l).nonzero(as_tuple=True)[0]  # samples whose lower level is l
        if lo.numel() == 0:
            continue
        c0 = _bilinear(levels[l], texcoords[lo])
        c1 = _bilinear(levels[min(l + 1, n - 1)], texcoords[lo])
        out = out.index_put((lo,), c0 + frac[lo] * (c1 - c0))
    return out


@torch.no_grad()
def texture_lod(texcoords: Tensor, texcoords_dx: Tensor, texcoords_dy: Tensor, texture_size) -> Tensor:
    """
    Mipmap level of detail from the texture coordinates at a pixel center and
    at its right (dx) and lower (dy) neighbors (evaluated on the same
    triangle, like the derivatives in a GPU pixel quad):

        lod = log2(max(|dt/dx|, |dt/dy|))   in texels per pixel

    texture_size: (Ht, Wt). Returns [M].
    """
    Ht, Wt = texture_size
    scale = texcoords.new_tensor([Wt, Ht])
    dx = ((texcoords_dx - texcoords) * scale).norm(dim=-1)
    dy = ((texcoords_dy - texcoords) * scale).norm(dim=-1)
    rho = torch.maximum(dx, dy).clamp(min=1e-12)
    return torch.log2(rho).clamp(min=0.0)
