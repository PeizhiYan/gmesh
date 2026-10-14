//
// Peizhi Yan
// Copyright (C) 2025
//
// Forward pass of 3DGS rasterization with a per-pixel depth limit (z-buffer).
//
// Adapted from gsplat v1.5.2 (gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu,
// Apache-2.0). The only behavioral change is the depth test: Gaussians are
// sorted front-to-back within each tile, so once a Gaussian is at or behind
// the depth limit of a pixel (e.g. the mesh z-buffer), it and all Gaussians
// after it are occluded, and the pixel stops accumulating.
//
// Because the pixel stops early, `last_ids` points at the last Gaussian that
// is in front of the depth limit. gsplat's backward kernel only walks
// Gaussians up to `last_ids`, so it can be reused unchanged.
//

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>

#include <tuple>

namespace cg = cooperative_groups;

namespace gmesh {

// Must match ALPHA_THRESHOLD in gsplat (gsplat/cuda/include/Common.h), so that
// gsplat's backward kernel re-derives exactly the same contributing Gaussians.
#define GMESH_ALPHA_THRESHOLD (1.f / 255.f)

template <uint32_t CDIM>
__global__ void rasterize_to_pixels_zbuffer_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const float2 *__restrict__ means2d,      // [C, N, 2]
    const float3 *__restrict__ conics,       // [C, N, 3]
    const float *__restrict__ colors,        // [C, N, CDIM]
    const float *__restrict__ opacities,     // [C, N]
    const float *__restrict__ depths,        // [C, N]
    const float *__restrict__ depth_limits,  // [C, image_height, image_width]
    const float depth_bias,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    float *__restrict__ render_colors, // [C, image_height, image_width, CDIM]
    float *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    int32_t *__restrict__ last_ids     // [C, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    depth_limits += camera_id * image_height * image_width;
    render_colors += camera_id * image_height * image_width * CDIM;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // Gaussians at or behind this depth are occluded at this pixel
    const float depth_limit = inside ? depth_limits[pix_id] + depth_bias : 0.f;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    float3 *xy_opacity_batch =
        reinterpret_cast<float3 *>(&id_batch[block_size]); // [block_size]
    float3 *conic_batch =
        reinterpret_cast<float3 *>(&xy_opacity_batch[block_size]); // [block_size]
    float *depth_batch =
        reinterpret_cast<float *>(&conic_batch[block_size]); // [block_size]

    // current visibility left to render
    float T = 1.0f;
    // index (into flatten_ids) of the most recent gaussian to write to this
    // thread's pixel. It starts *before* this tile's range, so that gsplat's
    // backward kernel sees "no Gaussian contributed" for pixels where every
    // Gaussian is occluded. The caller guarantees range_start >= 1.
    int32_t cur_idx = range_start - 1;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    float pix_out[CDIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N]
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = make_float3(xy.x, xy.y, opac);
            conic_batch[tr] = conics[g];
            depth_batch[tr] = depths[g];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            // z-test: Gaussians are sorted front-to-back, so this one and all
            // the following ones are occluded at this pixel
            if (depth_batch[t] >= depth_limit) {
                done = true;
                break;
            }

            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 delta = make_float2(xy_opac.x - px, xy_opac.y - py);
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < GMESH_ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] = pix_out[k];
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = cur_idx;
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_zbuffer_fwd_kernel(
    const at::Tensor &means2d,      // [C, N, 2]
    const at::Tensor &conics,       // [C, N, 3]
    const at::Tensor &colors,       // [C, N, CDIM]
    const at::Tensor &opacities,    // [C, N]
    const at::Tensor &depths,       // [C, N]
    const at::Tensor &depth_limits, // [C, image_height, image_width]
    const float depth_bias,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    at::Tensor &renders,            // [C, image_height, image_width, CDIM]
    at::Tensor &alphas,             // [C, image_height, image_width, 1]
    at::Tensor &last_ids            // [C, image_height, image_width]
) {
    uint32_t C = tile_offsets.size(0); // number of cameras
    uint32_t N = means2d.size(1);      // number of gaussians
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    if (C == 0 || tile_height == 0 || tile_width == 0) {
        return;
    }

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(float3) + sizeof(float3) + sizeof(float));

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_zbuffer_fwd_kernel<CDIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_zbuffer_fwd_kernel<CDIM>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            reinterpret_cast<const float2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<const float3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            depths.data_ptr<float>(),
            depth_limits.data_ptr<float>(),
            depth_bias,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>()
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    const uint32_t C = tile_offsets.size(0);
    const uint32_t channels = colors.size(-1);

    at::Tensor renders = at::empty(
        {C, image_height, image_width, channels}, means2d.options()
    );
    at::Tensor alphas =
        at::empty({C, image_height, image_width, 1}, means2d.options());
    at::Tensor last_ids = at::empty(
        {C, image_height, image_width}, means2d.options().dtype(at::kInt)
    );

#define __LAUNCH_KERNEL__(CDIM)                                                \
    case CDIM:                                                                 \
        launch_rasterize_to_pixels_zbuffer_fwd_kernel<CDIM>(                   \
            means2d,                                                           \
            conics,                                                            \
            colors,                                                            \
            opacities,                                                         \
            depths,                                                            \
            depth_limits,                                                      \
            static_cast<float>(depth_bias),                                    \
            static_cast<uint32_t>(image_width),                                \
            static_cast<uint32_t>(image_height),                               \
            static_cast<uint32_t>(tile_size),                                  \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            renders,                                                           \
            alphas,                                                            \
            last_ids                                                           \
        );                                                                     \
        break;

    // Same channel counts that gsplat's backward kernel is instantiated for.
    // The Python wrapper pads colors to one of these.
    switch (channels) {
        __LAUNCH_KERNEL__(1)
        __LAUNCH_KERNEL__(2)
        __LAUNCH_KERNEL__(3)
        __LAUNCH_KERNEL__(4)
        __LAUNCH_KERNEL__(5)
        __LAUNCH_KERNEL__(8)
        __LAUNCH_KERNEL__(9)
        __LAUNCH_KERNEL__(16)
        __LAUNCH_KERNEL__(17)
        __LAUNCH_KERNEL__(32)
        __LAUNCH_KERNEL__(33)
        __LAUNCH_KERNEL__(64)
        __LAUNCH_KERNEL__(65)
        __LAUNCH_KERNEL__(128)
        __LAUNCH_KERNEL__(129)
        __LAUNCH_KERNEL__(256)
        __LAUNCH_KERNEL__(257)
        __LAUNCH_KERNEL__(512)
        __LAUNCH_KERNEL__(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
#undef __LAUNCH_KERNEL__

    return std::make_tuple(renders, alphas, last_ids);
}

} // namespace gmesh
