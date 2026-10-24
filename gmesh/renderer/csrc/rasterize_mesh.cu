//
// Peizhi Yan
// Copyright (C) 2025
//
// Tile-based triangle visibility (z-buffer) for the mesh rasterizer.
//
// This kernel only decides WHICH triangle is the nearest at each pixel center.
// That decision is not differentiable; everything differentiable (barycentric
// coordinates, depth, attribute interpolation) is recomputed in PyTorch from
// the returned face ids, so autograd provides the gradients.
//
// Triangles are binned into tiles on the PyTorch side (same tile layout as
// gsplat). Each block handles one tile, each thread one pixel.
//

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>

#include <tuple>

namespace cg = cooperative_groups;

namespace gmesh {

// Twice the signed area of (a, b, p). The endpoints of edge (a, b) are put in
// a canonical order, so two triangles sharing an edge get bit-exact opposite
// values: no cracks along shared edges (watertight rasterization).
__device__ inline float edge_function(
    float ax, float ay, float bx, float by, const float px, const float py
) {
    const bool swap = (ax > bx) || (ax == bx && ay > by);
    if (swap) {
        float tx = ax, ty = ay;
        ax = bx;
        ay = by;
        bx = tx;
        by = ty;
    }
    const float e = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    return swap ? -e : e;
}

__global__ void rasterize_mesh_fwd_kernel(
    const uint32_t C,
    const uint32_t F,
    const uint32_t n_isects,
    const float *__restrict__ face_data, // [C, F, 9]: u0 v0 u1 v1 u2 v2 iz0 iz1 iz2
    const float inv_far,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects], indices in [C * F]
    int32_t *__restrict__ face_ids,           // [C, image_height, image_width]
    float *__restrict__ depths                // [C, image_height, image_width]
) {
    auto block = cg::this_thread_block();
    const int32_t camera_id = block.group_index().x;
    const int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i =
        block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j =
        block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    face_ids += camera_id * image_height * image_width;
    depths += camera_id * image_height * image_width;

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = i * image_width + j;
    const bool inside = (i < image_height && j < image_width);

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const int32_t block_size = block.size();
    const int32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;                          // [block_size]
    float *data_batch = (float *)&id_batch[block_size];        // [block_size * 9]

    const int32_t tr = block.thread_rank();

    // nearest fragment so far: largest inverse depth
    float best_iz = 0.f;
    int32_t best_id = -1;

    for (int32_t b = 0; b < num_batches; ++b) {
        block.sync();
        const int32_t batch_start = range_start + block_size * b;
        const int32_t idx = batch_start + tr;
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                data_batch[tr * 9 + k] = face_data[g * 9 + k];
            }
        }
        block.sync();

        if (!inside) {
            continue;
        }
        const int32_t batch_size = min(block_size, range_end - batch_start);
        for (int32_t t = 0; t < batch_size; ++t) {
            const float *fd = data_batch + t * 9;
            const float w0 = edge_function(fd[2], fd[3], fd[4], fd[5], px, py);
            const float w1 = edge_function(fd[4], fd[5], fd[0], fd[1], px, py);
            const float w2 = edge_function(fd[0], fd[1], fd[2], fd[3], px, py);
            const float area = w0 + w1 + w2;
            bool covered;
            if (area > 0.f) {
                covered = (w0 >= 0.f) && (w1 >= 0.f) && (w2 >= 0.f);
            } else if (area < 0.f) {
                covered = (w0 <= 0.f) && (w1 <= 0.f) && (w2 <= 0.f);
            } else {
                covered = false; // degenerate triangle
            }
            if (!covered) {
                continue;
            }
            // inverse depth is affine in screen space
            const float iz = (w0 * fd[6] + w1 * fd[7] + w2 * fd[8]) / area;
            if (!(iz >= inv_far)) { // beyond the far plane (or NaN)
                continue;
            }
            const int32_t f = id_batch[t] - camera_id * (int32_t)F;
            // nearest wins; ties go to the smaller face id (order independent)
            if (best_id < 0 || iz > best_iz || (iz == best_iz && f < best_id)) {
                best_iz = iz;
                best_id = f;
            }
        }
    }

    if (inside) {
        face_ids[pix_id] = best_id;
        depths[pix_id] = best_id >= 0 ? 1.f / best_iz : 0.f;
    }
}

std::tuple<at::Tensor, at::Tensor> rasterize_mesh_fwd(
    const at::Tensor &face_data,    // [C, F, 9]
    const double far_plane,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_size,
    const at::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(face_data));

    const uint32_t C = tile_offsets.size(0);
    const uint32_t F = face_data.size(1);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width = tile_offsets.size(2);
    const uint32_t n_isects = flatten_ids.size(0);

    at::Tensor face_ids = at::empty(
        {C, image_height, image_width}, face_data.options().dtype(at::kInt)
    );
    at::Tensor depths =
        at::empty({C, image_height, image_width}, face_data.options());

    if (C == 0 || tile_height == 0 || tile_width == 0) {
        return std::make_tuple(face_ids, depths);
    }

    const uint32_t ts = static_cast<uint32_t>(tile_size);
    dim3 threads = {ts, ts, 1};
    dim3 grid = {C, tile_height, tile_width};
    const int64_t shmem_size = ts * ts * (sizeof(int32_t) + 9 * sizeof(float));

    if (cudaFuncSetAttribute(
            rasterize_mesh_fwd_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_mesh_fwd_kernel<<<
        grid,
        threads,
        shmem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        C,
        F,
        n_isects,
        face_data.data_ptr<float>(),
        static_cast<float>(1.0 / far_plane),
        static_cast<uint32_t>(image_width),
        static_cast<uint32_t>(image_height),
        ts,
        tile_width,
        tile_height,
        tile_offsets.data_ptr<int32_t>(),
        flatten_ids.data_ptr<int32_t>(),
        face_ids.data_ptr<int32_t>(),
        depths.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(face_ids, depths);
}

} // namespace gmesh
