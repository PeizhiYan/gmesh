//
// Adapted from gsplat v1.5.2 (https://github.com/nerfstudio-project/gsplat),
// file gsplat/cuda/csrc/SphericalHarmonics.h.
// Copyright the gsplat authors. Licensed under the Apache License, Version 2.0
// (see LICENSE in this directory).
//
// Modified for Gmesh (Peizhi Yan, 2026):
//   - renamed the namespace gsplat -> gmesh_gsplat
//

#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gmesh_gsplat {

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    // outputs
    at::Tensor colors // [..., 2]
);

void launch_spherical_harmonics_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
    // outputs
    at::Tensor v_coeffs,            // [..., K, 3]
    at::optional<at::Tensor> v_dirs // [..., 3]
);

} // namespace gmesh_gsplat