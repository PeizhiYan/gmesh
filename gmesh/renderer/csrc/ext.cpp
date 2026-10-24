//
// Peizhi Yan
// Copyright (C) 2025
//

#include <torch/extension.h>

#include <tuple>

namespace gmesh {

std::tuple<at::Tensor, at::Tensor> rasterize_mesh_fwd(
    const at::Tensor &face_data,    // [C, F, 9]
    const double far_plane,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_size,
    const at::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_hybrid_fwd(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::Tensor &planes,
    const at::Tensor &depth_limits,
    const at::Tensor &coverages,
    const double softness,
    const bool skip_covered_back,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_hybrid_bwd(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::Tensor &planes,
    const at::Tensor &depth_limits,
    const at::Tensor &coverages,
    const double softness,
    const bool skip_covered_back,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    const at::Tensor &T_front,
    const at::Tensor &T_back,
    const at::Tensor &last_ids,
    const at::Tensor &v_out_front,
    const at::Tensor &v_T_front,
    const at::Tensor &v_out_back,
    const at::Tensor &v_T_back
);

} // namespace gmesh

// Gaussian projection, tile intersection and spherical harmonics, adapted from
// gsplat (Apache-2.0); see ./gsplat/GsplatOps.cpp
namespace gmesh_gsplat {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> projection_fwd(
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor opacities,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const int64_t image_width,
    const int64_t image_height,
    const double eps2d,
    const double near_plane,
    const double far_plane,
    const double radius_clip
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> projection_bwd(
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const int64_t image_width,
    const int64_t image_height,
    const double eps2d,
    const at::Tensor radii,
    const at::Tensor conics,
    const at::Tensor v_means2d,
    const at::Tensor v_depths,
    const at::Tensor v_conics,
    const bool viewmats_requires_grad
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> isect_tiles(
    const at::Tensor means2d,
    const at::Tensor radii,
    const at::Tensor depths,
    const int64_t tile_size,
    const int64_t tile_width,
    const int64_t tile_height,
    const bool sort
);

at::Tensor isect_offset_encode(
    const at::Tensor isect_ids,
    const int64_t C,
    const int64_t tile_width,
    const int64_t tile_height
);

at::Tensor sh_fwd(
    const int64_t degrees_to_use,
    const at::Tensor dirs,
    const at::Tensor coeffs,
    const at::optional<at::Tensor> masks
);

std::tuple<at::Tensor, at::Tensor> sh_bwd(
    const int64_t K,
    const int64_t degrees_to_use,
    const at::Tensor dirs,
    const at::Tensor coeffs,
    const at::optional<at::Tensor> masks,
    const at::Tensor v_colors,
    const bool compute_v_dirs
);

} // namespace gmesh_gsplat

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x)                                                         \
    TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INT(x)                                                           \
    TORCH_CHECK(x.scalar_type() == at::kInt, #x " must be int32")

std::tuple<at::Tensor, at::Tensor> rasterize_mesh_fwd(
    const at::Tensor &face_data,
    const double far_plane,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids
) {
    CHECK_INPUT(face_data);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_FLOAT(face_data);
    CHECK_INT(tile_offsets);
    CHECK_INT(flatten_ids);
    TORCH_CHECK(face_data.dim() == 3 && face_data.size(2) == 9,
                "face_data must be [C, F, 9]");
    TORCH_CHECK(tile_offsets.dim() == 3 && tile_offsets.size(0) == face_data.size(0),
                "tile_offsets must be [C, tile_height, tile_width]");
    TORCH_CHECK(tile_offsets.size(1) * tile_size >= image_height &&
                    tile_offsets.size(2) * tile_size >= image_width,
                "tile_offsets does not cover the image");
    return gmesh::rasterize_mesh_fwd(
        face_data, far_plane, image_width, image_height, tile_size,
        tile_offsets, flatten_ids
    );
}

static void check_hybrid_inputs(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::Tensor &planes,
    const at::Tensor &depth_limits,
    const at::Tensor &coverages,
    const double softness,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids
) {
    for (const at::Tensor *t : {&means2d, &conics, &colors, &opacities, &planes,
                                &depth_limits, &coverages}) {
        CHECK_INPUT((*t));
        CHECK_FLOAT((*t));
    }
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INT(tile_offsets);
    CHECK_INT(flatten_ids);
    TORCH_CHECK(softness > 0.0, "softness must be positive");

    TORCH_CHECK(means2d.dim() == 3 && means2d.size(2) == 2,
                "means2d must be [C, N, 2] (packed mode is not supported)");
    const int64_t C = means2d.size(0);
    const int64_t N = means2d.size(1);
    TORCH_CHECK(conics.sizes() == at::IntArrayRef({C, N, 3}), "conics must be [C, N, 3]");
    TORCH_CHECK(colors.dim() == 3 && colors.size(0) == C && colors.size(1) == N,
                "colors must be [C, N, channels]");
    TORCH_CHECK(opacities.sizes() == at::IntArrayRef({C, N}), "opacities must be [C, N]");
    TORCH_CHECK(planes.sizes() == at::IntArrayRef({C, N, 4}), "planes must be [C, N, 4]");
    TORCH_CHECK(depth_limits.sizes() == at::IntArrayRef({C, image_height, image_width}),
                "depth_limits must be [C, image_height, image_width]");
    TORCH_CHECK(coverages.sizes() == at::IntArrayRef({C, image_height, image_width}),
                "coverages must be [C, image_height, image_width]");
    TORCH_CHECK(tile_offsets.dim() == 3 && tile_offsets.size(0) == C,
                "tile_offsets must be [C, tile_height, tile_width]");
    TORCH_CHECK(tile_offsets.size(1) * tile_size >= image_height &&
                    tile_offsets.size(2) * tile_size >= image_width,
                "tile_offsets does not cover the image");
    TORCH_CHECK((tile_size * tile_size) % 32 == 0,
                "tile_size * tile_size must be a multiple of 32");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_hybrid_fwd(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::Tensor &planes,
    const at::Tensor &depth_limits,
    const at::Tensor &coverages,
    const double softness,
    const bool skip_covered_back,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids
) {
    check_hybrid_inputs(
        means2d, conics, colors, opacities, planes, depth_limits, coverages,
        softness, image_width, image_height, tile_size, tile_offsets, flatten_ids
    );
    return gmesh::rasterize_hybrid_fwd(
        means2d, conics, colors, opacities, planes, depth_limits, coverages,
        softness, skip_covered_back, image_width, image_height, tile_size,
        tile_offsets, flatten_ids
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_hybrid_bwd(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::Tensor &planes,
    const at::Tensor &depth_limits,
    const at::Tensor &coverages,
    const double softness,
    const bool skip_covered_back,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    const at::Tensor &T_front,
    const at::Tensor &T_back,
    const at::Tensor &last_ids,
    const at::Tensor &v_out_front,
    const at::Tensor &v_T_front,
    const at::Tensor &v_out_back,
    const at::Tensor &v_T_back
) {
    check_hybrid_inputs(
        means2d, conics, colors, opacities, planes, depth_limits, coverages,
        softness, image_width, image_height, tile_size, tile_offsets, flatten_ids
    );
    for (const at::Tensor *t : {&T_front, &T_back, &v_out_front, &v_T_front,
                                &v_out_back, &v_T_back}) {
        CHECK_INPUT((*t));
        CHECK_FLOAT((*t));
    }
    CHECK_INPUT(last_ids);
    CHECK_INT(last_ids);
    TORCH_CHECK(v_out_front.sizes() == v_out_back.sizes() &&
                    v_out_front.size(-1) == colors.size(-1),
                "gradients of the color outputs have the wrong shape");
    return gmesh::rasterize_hybrid_bwd(
        means2d, conics, colors, opacities, planes, depth_limits, coverages,
        softness, skip_covered_back, image_width, image_height, tile_size,
        tile_offsets, flatten_ids, T_front, T_back, last_ids, v_out_front,
        v_T_front, v_out_back, v_T_back
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // adapted from gsplat (Apache-2.0), see ./gsplat
    m.def("projection_fwd", &gmesh_gsplat::projection_fwd,
          "EWA projection of 3D Gaussians (pinhole camera), forward");
    m.def("projection_bwd", &gmesh_gsplat::projection_bwd,
          "EWA projection of 3D Gaussians (pinhole camera), backward");
    m.def("isect_tiles", &gmesh_gsplat::isect_tiles,
          "Tile intersections of the projected Gaussians, sorted by tile and depth");
    m.def("isect_offset_encode", &gmesh_gsplat::isect_offset_encode,
          "Start offset of each tile in the sorted intersections");
    m.def("sh_fwd", &gmesh_gsplat::sh_fwd, "Spherical harmonics, forward");
    m.def("sh_bwd", &gmesh_gsplat::sh_bwd, "Spherical harmonics, backward");

    m.def(
        "rasterize_mesh_fwd",
        &rasterize_mesh_fwd,
        "Tile-based triangle visibility (nearest face id per pixel)"
    );
    m.def(
        "rasterize_hybrid_fwd",
        &rasterize_hybrid_fwd,
        "Hybrid Gaussian/mesh compositing with a soft per-ray depth test (forward)"
    );
    m.def(
        "rasterize_hybrid_bwd",
        &rasterize_hybrid_bwd,
        "Hybrid Gaussian/mesh compositing with a soft per-ray depth test (backward)"
    );
}
