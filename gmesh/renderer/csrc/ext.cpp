//
// Peizhi Yan
// Copyright (C) 2025
//

#include <torch/extension.h>

#include <tuple>

namespace gmesh {

std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_zbuffer_fwd(
    const at::Tensor &means2d,      // [C, N, 2]
    const at::Tensor &conics,       // [C, N, 3]
    const at::Tensor &colors,       // [C, N, channels]
    const at::Tensor &opacities,    // [C, N]
    const at::Tensor &depths,       // [C, N]
    const at::Tensor &depth_limits, // [C, image_height, image_width]
    const double depth_bias,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_size,
    const at::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
);

} // namespace gmesh

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

std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_zbuffer_fwd(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::Tensor &depths,
    const at::Tensor &depth_limits,
    const double depth_bias,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids
) {
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(depths);
    CHECK_INPUT(depth_limits);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_FLOAT(means2d);
    CHECK_FLOAT(conics);
    CHECK_FLOAT(colors);
    CHECK_FLOAT(opacities);
    CHECK_FLOAT(depths);
    CHECK_FLOAT(depth_limits);
    CHECK_INT(tile_offsets);
    CHECK_INT(flatten_ids);

    TORCH_CHECK(means2d.dim() == 3 && means2d.size(2) == 2,
                "means2d must be [C, N, 2] (packed mode is not supported)");
    const int64_t C = means2d.size(0);
    const int64_t N = means2d.size(1);
    TORCH_CHECK(conics.dim() == 3 && conics.size(0) == C &&
                    conics.size(1) == N && conics.size(2) == 3,
                "conics must be [C, N, 3]");
    TORCH_CHECK(colors.dim() == 3 && colors.size(0) == C &&
                    colors.size(1) == N,
                "colors must be [C, N, channels]");
    TORCH_CHECK(opacities.dim() == 2 && opacities.size(0) == C &&
                    opacities.size(1) == N,
                "opacities must be [C, N]");
    TORCH_CHECK(depths.dim() == 2 && depths.size(0) == C &&
                    depths.size(1) == N,
                "depths must be [C, N]");
    TORCH_CHECK(depth_limits.dim() == 3 && depth_limits.size(0) == C &&
                    depth_limits.size(1) == image_height &&
                    depth_limits.size(2) == image_width,
                "depth_limits must be [C, image_height, image_width]");
    TORCH_CHECK(tile_offsets.dim() == 3 && tile_offsets.size(0) == C,
                "tile_offsets must be [C, tile_height, tile_width]");
    TORCH_CHECK(tile_offsets.size(1) * tile_size >= image_height &&
                    tile_offsets.size(2) * tile_size >= image_width,
                "tile_offsets does not cover the image");

    return gmesh::rasterize_to_pixels_zbuffer_fwd(
        means2d,
        conics,
        colors,
        opacities,
        depths,
        depth_limits,
        depth_bias,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rasterize_to_pixels_zbuffer_fwd",
        &rasterize_to_pixels_zbuffer_fwd,
        "3DGS rasterization (forward) with a per-pixel depth limit"
    );
}
