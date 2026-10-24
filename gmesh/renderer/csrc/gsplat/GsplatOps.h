//
// Adapted from gsplat v1.5.2 (https://github.com/nerfstudio-project/gsplat),
// files gsplat/cuda/csrc/Projection.cpp, Intersect.cpp and SphericalHarmonics.cpp.
// Copyright the gsplat authors. Licensed under the Apache License, Version 2.0
// (see LICENSE in this directory).
//
// Modified for Gmesh (Peizhi Yan, 2026):
//   - renamed the namespace gsplat -> gmesh_gsplat
//   - kept only the host functions of the fused (non-packed) 3DGS projection,
//     the tile intersection and the spherical harmonics, in one file
//   - declarations of the host functions in GsplatOps.cpp
//

#pragma once

#include <ATen/core/Tensor.h>
#include <tuple>

#include "Common.h"

namespace gmesh_gsplat {

// gsplat host functions (unchanged signatures)

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_fused_fwd(
    const at::Tensor means,
    const at::optional<at::Tensor> covars,
    const at::optional<at::Tensor> quats,
    const at::optional<at::Tensor> scales,
    const at::optional<at::Tensor> opacities,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_fused_bwd(
    const at::Tensor means,
    const at::optional<at::Tensor> covars,
    const at::optional<at::Tensor> quats,
    const at::optional<at::Tensor> scales,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    const at::Tensor radii,
    const at::Tensor conics,
    const at::optional<at::Tensor> compensations,
    const at::Tensor v_means2d,
    const at::Tensor v_depths,
    const at::Tensor v_conics,
    const at::optional<at::Tensor> v_compensations,
    const bool viewmats_requires_grad
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor means2d,
    const at::Tensor radii,
    const at::Tensor depths,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort
);

at::Tensor intersect_offset(
    const at::Tensor isect_ids,
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
);

at::Tensor spherical_harmonics_fwd(
    const uint32_t degrees_to_use,
    const at::Tensor dirs,
    const at::Tensor coeffs,
    const at::optional<at::Tensor> masks
);

std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
    const uint32_t K,
    const uint32_t degrees_to_use,
    const at::Tensor dirs,
    const at::Tensor coeffs,
    const at::optional<at::Tensor> masks,
    const at::Tensor v_colors,
    bool compute_v_dirs
);

} // namespace gmesh_gsplat
