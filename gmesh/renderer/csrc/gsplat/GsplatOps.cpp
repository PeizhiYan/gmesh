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
//   - added pinhole-camera entry points used by Gmesh (end of the file)
//

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"             // where all the macros are defined
#include "Intersect.h"          // launch functions of the tile intersection
#include "Projection.h"         // launch functions of the projection
#include "SphericalHarmonics.h" // launch functions of the spherical harmonics
#include "GsplatOps.h"

namespace gmesh_gsplat {

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dgs_fused_fwd(
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6] optional
    const at::optional<at::Tensor> quats,  // [N, 4] optional
    const at::optional<at::Tensor> scales, // [N, 3] optional
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    at::Tensor radii = at::empty({C, N, 2}, means.options().dtype(at::kInt));
    at::Tensor means2d = at::empty({C, N, 2}, means.options());
    at::Tensor depths = at::empty({C, N}, means.options());
    at::Tensor conics = at::empty({C, N, 3}, means.options());
    at::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = at::zeros({C, N}, means.options());
    }

    launch_projection_ewa_3dgs_fused_fwd_kernel(
        // inputs
        means,
        covars,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        camera_model,
        // outputs
        radii,
        means2d,
        depths,
        conics,
        calc_compensations ? at::optional<at::Tensor>(compensations)
                           : c10::nullopt
    );
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_fused_bwd(
    // fwd inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6] optional
    const at::optional<at::Tensor> quats,  // [N, 4] optional
    const at::optional<at::Tensor> scales, // [N, 3] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [C, N, 2]
    const at::Tensor conics,                      // [C, N, 3]
    const at::optional<at::Tensor> compensations, // [C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [C, N, 2]
    const at::Tensor v_depths,                      // [C, N]
    const at::Tensor v_conics,                      // [C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [C, N] optional
    const bool viewmats_requires_grad
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }

    at::Tensor v_means = at::zeros_like(means);
    at::Tensor v_covars, v_quats, v_scales; // optional
    if (covars.has_value()) {
        v_covars = at::zeros_like(covars.value());
    } else {
        v_quats = at::zeros_like(quats.value());
        v_scales = at::zeros_like(scales.value());
    }
    at::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = at::zeros_like(viewmats);
    }

    launch_projection_ewa_3dgs_fused_bwd_kernel(
        // inputs
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        camera_model,
        radii,
        conics,
        compensations,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
        viewmats_requires_grad,
        // outputs
        v_means,
        v_covars,
        v_quats,
        v_scales,
        v_viewmats
    );

    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor means2d,                    // [C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [C, N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [C, N] or [nnz]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);

    uint32_t n_elements = means2d.numel() / 2;
    bool packed = means2d.dim() == 2;
    if (packed) {
        TORCH_CHECK(
            camera_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, camera_ids and gaussian_ids must be provided."
        );
        CHECK_INPUT(camera_ids.value());
        CHECK_INPUT(gaussian_ids.value());
    }

    uint32_t n_tiles = tile_width * tile_height;
    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    // first pass: compute number of tiles per gaussian
    at::Tensor tiles_per_gauss =
        at::empty_like(depths, depths.options().dtype(at::kInt));
    int64_t n_isects;
    at::Tensor cum_tiles_per_gauss;
    if (n_elements) {
        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            packed ? camera_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            C,
            tile_size,
            tile_width,
            tile_height,
            c10::nullopt, // cum_tiles_per_gauss
            // outputs
            at::optional<at::Tensor>(tiles_per_gauss),
            c10::nullopt, // isect_ids
            c10::nullopt  // flatten_ids
        );
        cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    at::Tensor isect_ids =
        at::empty({n_isects}, depths.options().dtype(at::kLong));
    at::Tensor flatten_ids =
        at::empty({n_isects}, depths.options().dtype(at::kInt));
    if (n_isects) {
        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            packed ? camera_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            C,
            tile_size,
            tile_width,
            tile_height,
            cum_tiles_per_gauss,
            // outputs
            c10::nullopt, // tiles_per_gauss
            at::optional<at::Tensor>(isect_ids),
            at::optional<at::Tensor>(flatten_ids)
        );
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
        radix_sort_double_buffer(
            n_isects,
            tile_n_bits,
            cam_n_bits,
            isect_ids,
            flatten_ids,
            isect_ids_sorted,
            flatten_ids_sorted
        );
        return std::make_tuple(
            tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted
        );
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

at::Tensor intersect_offset(
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    DEVICE_GUARD(isect_ids);
    CHECK_INPUT(isect_ids);

    at::Tensor offsets = at::empty(
        {C, tile_height, tile_width}, isect_ids.options().dtype(at::kInt)
    );
    launch_intersect_offset_kernel(
        isect_ids, C, tile_width, tile_height, offsets
    );
    return offsets;
}

at::Tensor spherical_harmonics_fwd(
    const uint32_t degrees_to_use,
    const at::Tensor dirs,               // [..., 3]
    const at::Tensor coeffs,             // [..., K, 3]
    const at::optional<at::Tensor> masks // [...]
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");

    at::Tensor colors = at::empty_like(dirs); // [..., 3]

    launch_spherical_harmonics_fwd_kernel(
        degrees_to_use, dirs, coeffs, masks, colors
    );
    return colors; // [..., 3]
}

std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
    const uint32_t K,
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
    bool compute_v_dirs
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    CHECK_INPUT(v_colors);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(v_colors.size(-1) == 3, "v_colors must have last dimension 3");
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    const uint32_t N = dirs.numel() / 3;

    at::Tensor v_coeffs = at::zeros_like(coeffs);
    at::Tensor v_dirs;
    if (compute_v_dirs) {
        v_dirs = at::zeros_like(dirs);
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    uint32_t n_elements = N;
    uint32_t shmem_size = 0;
    launch_spherical_harmonics_bwd_kernel(
        degrees_to_use,
        dirs,
        coeffs,
        masks,
        v_colors,
        v_coeffs,
        v_dirs.defined() ? at::optional<at::Tensor>(v_dirs) : c10::nullopt
    );
    return std::make_tuple(v_coeffs, v_dirs); // [..., K, 3], [..., 3]
}

//
// Gmesh entry points (pinhole camera, Gaussians given by quaternions and scales,
// no opacity compensation), bound to Python in ../ext.cpp
//

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> projection_fwd(
    const at::Tensor means,     // [N, 3]
    const at::Tensor quats,     // [N, 4]
    const at::Tensor scales,    // [N, 3]
    const at::Tensor opacities, // [N]
    const at::Tensor viewmats,  // [C, 4, 4]
    const at::Tensor Ks,        // [C, 3, 3]
    const int64_t image_width,
    const int64_t image_height,
    const double eps2d,
    const double near_plane,
    const double far_plane,
    const double radius_clip
) {
    CHECK_INPUT(opacities);
    // returns radii, means2d, depths, conics
    auto out = projection_ewa_3dgs_fused_fwd(
        means, c10::nullopt, quats, scales, opacities, viewmats, Ks,
        (uint32_t)image_width, (uint32_t)image_height, (float)eps2d,
        (float)near_plane, (float)far_plane, (float)radius_clip,
        false, CameraModelType::PINHOLE
    );
    return std::make_tuple(
        std::get<0>(out), std::get<1>(out), std::get<2>(out), std::get<3>(out)
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> projection_bwd(
    const at::Tensor means,     // [N, 3]
    const at::Tensor quats,     // [N, 4]
    const at::Tensor scales,    // [N, 3]
    const at::Tensor viewmats,  // [C, 4, 4]
    const at::Tensor Ks,        // [C, 3, 3]
    const int64_t image_width,
    const int64_t image_height,
    const double eps2d,
    const at::Tensor radii,     // [C, N, 2]
    const at::Tensor conics,    // [C, N, 3]
    const at::Tensor v_means2d, // [C, N, 2]
    const at::Tensor v_depths,  // [C, N]
    const at::Tensor v_conics,  // [C, N, 3]
    const bool viewmats_requires_grad
) {
    // returns v_means, v_quats, v_scales, v_viewmats
    auto out = projection_ewa_3dgs_fused_bwd(
        means, c10::nullopt, quats, scales, viewmats, Ks,
        (uint32_t)image_width, (uint32_t)image_height, (float)eps2d,
        CameraModelType::PINHOLE, radii, conics, c10::nullopt,
        v_means2d, v_depths, v_conics, c10::nullopt, viewmats_requires_grad
    );
    return std::make_tuple(
        std::get<0>(out), std::get<2>(out), std::get<3>(out), std::get<4>(out)
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> isect_tiles(
    const at::Tensor means2d, // [C, N, 2]
    const at::Tensor radii,   // [C, N, 2]
    const at::Tensor depths,  // [C, N]
    const int64_t tile_size,
    const int64_t tile_width,
    const int64_t tile_height,
    const bool sort
) {
    TORCH_CHECK(means2d.dim() == 3, "means2d must be [C, N, 2] (packed mode is not supported)");
    // returns tiles_per_gauss, isect_ids, flatten_ids
    return intersect_tile(
        means2d, radii, depths, c10::nullopt, c10::nullopt,
        (uint32_t)means2d.size(0), (uint32_t)tile_size, (uint32_t)tile_width,
        (uint32_t)tile_height, sort
    );
}

at::Tensor isect_offset_encode(
    const at::Tensor isect_ids, // [n_isects]
    const int64_t C,
    const int64_t tile_width,
    const int64_t tile_height
) {
    return intersect_offset(isect_ids, (uint32_t)C, (uint32_t)tile_width, (uint32_t)tile_height);
}

at::Tensor sh_fwd(
    const int64_t degrees_to_use,
    const at::Tensor dirs,               // [..., 3]
    const at::Tensor coeffs,             // [..., K, 3]
    const at::optional<at::Tensor> masks // [...]
) {
    return spherical_harmonics_fwd((uint32_t)degrees_to_use, dirs, coeffs, masks);
}

std::tuple<at::Tensor, at::Tensor> sh_bwd(
    const int64_t K,
    const int64_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
    const bool compute_v_dirs
) {
    return spherical_harmonics_bwd(
        (uint32_t)K, (uint32_t)degrees_to_use, dirs, coeffs, masks, v_colors, compute_v_dirs
    );
}

} // namespace gmesh_gsplat
