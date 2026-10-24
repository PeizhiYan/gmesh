//
// Peizhi Yan
// Copyright (C) 2025
//
// Hybrid Gaussian / mesh compositing with a soft, per-pixel depth test.
//
// Portions adapted from gsplat v1.5.2 (https://github.com/nerfstudio-project/gsplat),
// files gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu and RasterizeToPixels3DGSBwd.cu:
// the tile/batch structure, the front-to-back alpha compositing and its backward
// pass. Copyright the gsplat authors, licensed under the Apache License,
// Version 2.0 (see ./gsplat/LICENSE). Modified for Gmesh: two compositing layers
// (in front of / behind the mesh) with the soft depth test described below, the
// gradients of the depth test, and no background / masks / absgrad.
//
// Each Gaussian's depth at pixel p uses the same local affine (EWA)
// approximation as its 2D splat: the depth z conditioned on the projected
// position (u, v) = p is Gaussian with
//
//     z*(p) = z_c + g . (p - mean2d),   std sigma
//
// (g, sigma from the projected 3D covariance, computed in PyTorch). With the
// mesh depth D(p), the fraction of the Gaussian in front of the mesh is
//
//     f = Phi(s),   s = (D - z*) / (sigma * softness)
//
// The front part (alpha * f) is composited into a "front" layer, the back
// part (alpha * (1 - f)) into a "back" layer (each front-to-back in the
// usual 3DGS sorted order). The caller composes the final pixel as
//
//     C = C_front + T_front * (a_m * c_mesh + (1 - a_m) * (C_back + T_back * bg))
//
// Pixels without mesh (D = +inf) have f = 1 and an empty back layer, i.e.
// plain 3DGS. The backward pass gives gradients for all Gaussian quantities,
// including the depth plane (z_c, g, sigma), and for the mesh depth D.
//

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <tuple>

namespace cg = cooperative_groups;

namespace gmesh {

#define GMESH_HYBRID_ALPHA_THRESHOLD (1.f / 255.f)
#define GMESH_HYBRID_T_EPS 1e-4f

// depth plane of a Gaussian: [z_c, g_x, g_y, sigma]
// delta = mean2d - pixel center, so z* = z_c - g . delta
__device__ inline float gaussian_split_offset(
    const float *plane, const float2 delta, const float D, const float inv_softness
) {
    const float z_star = plane[0] - (plane[1] * delta.x + plane[2] * delta.y);
    return (D - z_star) * inv_softness / plane[3];
}

template <uint32_t CDIM>
__global__ void rasterize_hybrid_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const float2 *__restrict__ means2d,     // [C, N, 2]
    const float3 *__restrict__ conics,      // [C, N, 3]
    const float *__restrict__ colors,       // [C, N, CDIM]
    const float *__restrict__ opacities,    // [C, N]
    const float *__restrict__ planes,       // [C, N, 4] z_c, g_x, g_y, sigma
    const float *__restrict__ depth_limits, // [C, H, W] mesh depth, +inf if none
    const float *__restrict__ coverages,    // [C, H, W] mesh coverage
    const float inv_softness,
    const bool skip_covered_back,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    float *__restrict__ out_front,            // [C, H, W, CDIM]
    float *__restrict__ T_front,              // [C, H, W]
    float *__restrict__ out_back,             // [C, H, W, CDIM]
    float *__restrict__ T_back,               // [C, H, W]
    int32_t *__restrict__ last_ids            // [C, H, W, 2]
) {
    auto block = cg::this_thread_block();
    const int32_t camera_id = block.group_index().x;
    const int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i =
        block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j =
        block.group_index().z * tile_size + block.thread_index().x;

    const int32_t n_pix = image_height * image_width;
    tile_offsets += camera_id * tile_height * tile_width;
    depth_limits += camera_id * n_pix;
    coverages += camera_id * n_pix;
    out_front += camera_id * n_pix * CDIM;
    out_back += camera_id * n_pix * CDIM;
    T_front += camera_id * n_pix;
    T_back += camera_id * n_pix;
    last_ids += camera_id * n_pix * 2;

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = i * image_width + j;
    const bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    const float D = inside ? depth_limits[pix_id] : 0.f;
    const bool has_mesh = inside && isfinite(D);
    const float coverage = inside ? coverages[pix_id] : 0.f;
    bool back_active = has_mesh && !(skip_covered_back && coverage >= 1.f);

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const int32_t block_size = block.size();
    const int32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;                                              // [bs]
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[block_size]); // [bs]
    float3 *conic_batch = &xy_opacity_batch[block_size];                          // [bs]
    float *plane_batch = reinterpret_cast<float *>(&conic_batch[block_size]);     // [bs * 4]

    float T_f = 1.f, T_b = 1.f;
    float pix_f[CDIM] = {0.f};
    float pix_b[CDIM] = {0.f};
    // index of the last Gaussian that contributed to each layer
    int32_t last_f = range_start - 1;
    int32_t last_b = range_start - 1;

    const int32_t tr = block.thread_rank();

    for (int32_t b = 0; b < num_batches; ++b) {
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        const int32_t batch_start = range_start + block_size * b;
        const int32_t idx = batch_start + tr;
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_opacity_batch[tr] = make_float3(xy.x, xy.y, opacities[g]);
            conic_batch[tr] = conics[g];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                plane_batch[tr * 4 + k] = planes[g * 4 + k];
            }
        }
        block.sync();

        const int32_t batch_size = min(block_size, range_end - batch_start);
        for (int32_t t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 delta = make_float2(xy_opac.x - px, xy_opac.y - py);
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < GMESH_HYBRID_ALPHA_THRESHOLD) {
                continue;
            }

            // fraction of the Gaussian in front of the mesh at this pixel
            const float frac = has_mesh
                ? normcdff(gaussian_split_offset(plane_batch + t * 4, delta, D, inv_softness))
                : 1.f;

            const int32_t g = id_batch[t];
            const float *c_ptr = colors + g * CDIM;

            const float alpha_f = alpha * frac;
            if (alpha_f >= GMESH_HYBRID_ALPHA_THRESHOLD) {
                const float next_T = T_f * (1.f - alpha_f);
                if (next_T <= GMESH_HYBRID_T_EPS) {
                    // nothing behind is visible anymore (mesh and back layer included)
                    done = true;
                    break;
                }
                const float vis = alpha_f * T_f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    pix_f[k] += c_ptr[k] * vis;
                }
                T_f = next_T;
                last_f = batch_start + t;
            }

            if (back_active) {
                const float alpha_b = alpha * (1.f - frac);
                if (alpha_b >= GMESH_HYBRID_ALPHA_THRESHOLD) {
                    const float next_T = T_b * (1.f - alpha_b);
                    if (next_T <= GMESH_HYBRID_T_EPS) {
                        back_active = false;
                    } else {
                        const float vis = alpha_b * T_b;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            pix_b[k] += c_ptr[k] * vis;
                        }
                        T_b = next_T;
                        last_b = batch_start + t;
                    }
                }
            }
        }
    }

    if (inside) {
        T_front[pix_id] = T_f;
        T_back[pix_id] = T_b;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            out_front[pix_id * CDIM + k] = pix_f[k];
            out_back[pix_id * CDIM + k] = pix_b[k];
        }
        last_ids[pix_id * 2 + 0] = last_f;
        last_ids[pix_id * 2 + 1] = last_b;
    }
}

template <uint32_t CDIM>
__global__ void rasterize_hybrid_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const float2 *__restrict__ means2d,     // [C, N, 2]
    const float3 *__restrict__ conics,      // [C, N, 3]
    const float *__restrict__ colors,       // [C, N, CDIM]
    const float *__restrict__ opacities,    // [C, N]
    const float *__restrict__ planes,       // [C, N, 4]
    const float *__restrict__ depth_limits, // [C, H, W]
    const float *__restrict__ coverages,    // [C, H, W]
    const float inv_softness,
    const bool skip_covered_back,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // forward outputs
    const float *__restrict__ T_front,    // [C, H, W]
    const float *__restrict__ T_back,     // [C, H, W]
    const int32_t *__restrict__ last_ids, // [C, H, W, 2]
    // gradients of the outputs
    const float *__restrict__ v_out_front, // [C, H, W, CDIM]
    const float *__restrict__ v_T_front,   // [C, H, W]
    const float *__restrict__ v_out_back,  // [C, H, W, CDIM]
    const float *__restrict__ v_T_back,    // [C, H, W]
    // gradients of the inputs
    float2 *__restrict__ v_means2d,    // [C, N, 2]
    float3 *__restrict__ v_conics,     // [C, N, 3]
    float *__restrict__ v_colors,      // [C, N, CDIM]
    float *__restrict__ v_opacities,   // [C, N]
    float *__restrict__ v_planes,      // [C, N, 4]
    float *__restrict__ v_depth_limits // [C, H, W]
) {
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t camera_id = block.group_index().x;
    const int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i =
        block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j =
        block.group_index().z * tile_size + block.thread_index().x;

    const int32_t n_pix = image_height * image_width;
    tile_offsets += camera_id * tile_height * tile_width;
    depth_limits += camera_id * n_pix;
    coverages += camera_id * n_pix;
    T_front += camera_id * n_pix;
    T_back += camera_id * n_pix;
    last_ids += camera_id * n_pix * 2;
    v_out_front += camera_id * n_pix * CDIM;
    v_out_back += camera_id * n_pix * CDIM;
    v_T_front += camera_id * n_pix;
    v_T_back += camera_id * n_pix;
    v_depth_limits += camera_id * n_pix;

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const bool inside = (i < image_height && j < image_width);
    // clamp this value to the last pixel (for reads only)
    const int32_t pix_id = min((int32_t)(i * image_width + j), n_pix - 1);

    const float D = inside ? depth_limits[pix_id] : 0.f;
    const bool has_mesh = inside && isfinite(D);
    const float coverage = inside ? coverages[pix_id] : 0.f;
    const bool back_enabled = has_mesh && !(skip_covered_back && coverage >= 1.f);

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const int32_t block_size = block.size();
    const int32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[block_size]);
    float3 *conic_batch = &xy_opacity_batch[block_size];
    float *plane_batch = reinterpret_cast<float *>(&conic_batch[block_size]); // [bs * 4]
    float *rgbs_batch = &plane_batch[block_size * 4];                         // [bs * CDIM]

    // transmittance AFTER the last Gaussian of each layer
    const float T_f_final = inside ? T_front[pix_id] : 1.f;
    const float T_b_final = inside ? T_back[pix_id] : 1.f;
    float T_f = T_f_final, T_b = T_b_final;
    float buffer_f[CDIM] = {0.f};
    float buffer_b[CDIM] = {0.f};

    const int32_t last_f = inside ? last_ids[pix_id * 2 + 0] : -1;
    const int32_t last_b = inside ? last_ids[pix_id * 2 + 1] : -1;
    const int32_t bin_final = inside ? max(last_f, last_b) : -1;

    float v_cf[CDIM], v_cb[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_cf[k] = inside ? v_out_front[pix_id * CDIM + k] : 0.f;
        v_cb[k] = inside ? v_out_back[pix_id * CDIM + k] : 0.f;
    }
    const float v_Tf = inside ? v_T_front[pix_id] : 0.f;
    const float v_Tb = inside ? v_T_back[pix_id] : 0.f;
    float v_D = 0.f;

    const int32_t tr = block.thread_rank();
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

    for (int32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetches 1 Gaussian, from back to front
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            const int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            xy_opacity_batch[tr] = make_float3(xy.x, xy.y, opacities[g]);
            conic_batch[tr] = conics[g];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                plane_batch[tr * 4 + k] = planes[g * 4 + k];
            }
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        block.sync();

        // t = 0 is the furthest back Gaussian of the batch
        for (int32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            const int32_t cur = batch_end - t; // index in flatten_ids
            bool valid = inside && (cur <= bin_final);

            float alpha = 0.f, opac = 0.f, vis = 0.f;
            float2 delta = make_float2(0.f, 0.f);
            float3 conic = make_float3(0.f, 0.f, 0.f);
            if (valid) {
                conic = conic_batch[t];
                const float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = make_float2(xy_opac.x - px, xy_opac.y - py);
                const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < GMESH_HYBRID_ALPHA_THRESHOLD) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }

            float v_rgb_local[CDIM] = {0.f};
            float3 v_conic_local = make_float3(0.f, 0.f, 0.f);
            float2 v_xy_local = make_float2(0.f, 0.f);
            float v_opacity_local = 0.f;
            float v_plane_local[4] = {0.f, 0.f, 0.f, 0.f};

            if (valid) {
                const float *rgb = rgbs_batch + t * CDIM;
                const float *plane = plane_batch + t * 4;

                // recompute the split exactly as in the forward pass
                float frac = 1.f, s = 0.f;
                if (has_mesh) {
                    s = gaussian_split_offset(plane, delta, D, inv_softness);
                    frac = normcdff(s);
                }
                const float alpha_f = alpha * frac;
                const float alpha_b = alpha * (1.f - frac);
                const bool use_f =
                    (cur <= last_f) && (alpha_f >= GMESH_HYBRID_ALPHA_THRESHOLD);
                const bool use_b = back_enabled && (cur <= last_b) &&
                                   (alpha_b >= GMESH_HYBRID_ALPHA_THRESHOLD);

                float v_alpha_f = 0.f, v_alpha_b = 0.f;
                if (use_f) {
                    const float ra_f = 1.f / (1.f - alpha_f);
                    T_f *= ra_f; // transmittance before this Gaussian
                    const float fac = alpha_f * T_f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_rgb_local[k] += fac * v_cf[k];
                        v_alpha_f += (rgb[k] * T_f - buffer_f[k] * ra_f) * v_cf[k];
                        buffer_f[k] += rgb[k] * fac;
                    }
                    v_alpha_f += -T_f_final * ra_f * v_Tf;
                }
                if (use_b) {
                    const float ra_b = 1.f / (1.f - alpha_b);
                    T_b *= ra_b;
                    const float fac = alpha_b * T_b;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_rgb_local[k] += fac * v_cb[k];
                        v_alpha_b += (rgb[k] * T_b - buffer_b[k] * ra_b) * v_cb[k];
                        buffer_b[k] += rgb[k] * fac;
                    }
                    v_alpha_b += -T_b_final * ra_b * v_Tb;
                }

                // alpha_f = alpha * frac, alpha_b = alpha * (1 - frac)
                const float v_alpha = v_alpha_f * frac + v_alpha_b * (1.f - frac);
                const float v_frac = (v_alpha_f - v_alpha_b) * alpha;

                if (opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local = make_float3(
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    );
                    v_xy_local = make_float2(
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    );
                    v_opacity_local = vis * v_alpha;
                }

                if (has_mesh && v_frac != 0.f) {
                    // frac = Phi(s), s = (D - z*) / (sigma * softness),
                    // z* = z_c - g . delta, delta = mean2d - pixel
                    const float phi = 0.3989422804014327f * __expf(-0.5f * s * s);
                    const float v_s = v_frac * phi;
                    const float k = inv_softness / plane[3]; // ds/dD = -ds/dz*
                    v_D += v_s * k;
                    v_plane_local[0] = -v_s * k;             // z_c
                    v_plane_local[1] = v_s * k * delta.x;    // g_x
                    v_plane_local[2] = v_s * k * delta.y;    // g_y
                    v_plane_local[3] = -v_s * s / plane[3];  // sigma
                    v_xy_local.x += v_s * k * plane[1];      // through delta
                    v_xy_local.y += v_s * k * plane[2];
                }
            }

            // sum over the warp, then one atomic add per Gaussian
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                v_rgb_local[k] = cg::reduce(warp, v_rgb_local[k], cg::plus<float>());
            }
            v_conic_local.x = cg::reduce(warp, v_conic_local.x, cg::plus<float>());
            v_conic_local.y = cg::reduce(warp, v_conic_local.y, cg::plus<float>());
            v_conic_local.z = cg::reduce(warp, v_conic_local.z, cg::plus<float>());
            v_xy_local.x = cg::reduce(warp, v_xy_local.x, cg::plus<float>());
            v_xy_local.y = cg::reduce(warp, v_xy_local.y, cg::plus<float>());
            v_opacity_local = cg::reduce(warp, v_opacity_local, cg::plus<float>());
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                v_plane_local[k] = cg::reduce(warp, v_plane_local[k], cg::plus<float>());
            }

            if (warp.thread_rank() == 0) {
                const int32_t g = id_batch[t];
                float *v_rgb_ptr = v_colors + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }
                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);
                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);
                gpuAtomicAdd(v_opacities + g, v_opacity_local);
                float *v_plane_ptr = v_planes + 4 * g;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    gpuAtomicAdd(v_plane_ptr + k, v_plane_local[k]);
                }
            }
        }
    }

    if (inside) {
        v_depth_limits[pix_id] = v_D;
    }
}

namespace {

void check_hybrid_launch(const void *kernel, int64_t shmem_size) {
    if (cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }
}

int64_t hybrid_fwd_shmem(uint32_t tile_size) {
    return tile_size * tile_size *
           (sizeof(int32_t) + 2 * sizeof(float3) + 4 * sizeof(float));
}

int64_t hybrid_bwd_shmem(uint32_t tile_size, uint32_t cdim) {
    return tile_size * tile_size *
           (sizeof(int32_t) + 2 * sizeof(float3) + (4 + cdim) * sizeof(float));
}

} // namespace

template <uint32_t CDIM>
void launch_rasterize_hybrid_fwd(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::Tensor &planes,
    const at::Tensor &depth_limits,
    const at::Tensor &coverages,
    const float inv_softness,
    const bool skip_covered_back,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    at::Tensor &out_front,
    at::Tensor &T_front,
    at::Tensor &out_back,
    at::Tensor &T_back,
    at::Tensor &last_ids
) {
    const uint32_t C = tile_offsets.size(0);
    const uint32_t N = means2d.size(1);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width = tile_offsets.size(2);
    const uint32_t n_isects = flatten_ids.size(0);
    if (C == 0 || tile_height == 0 || tile_width == 0) {
        return;
    }

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};
    const int64_t shmem_size = hybrid_fwd_shmem(tile_size);
    check_hybrid_launch((const void *)rasterize_hybrid_fwd_kernel<CDIM>, shmem_size);

    rasterize_hybrid_fwd_kernel<CDIM>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            reinterpret_cast<const float2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<const float3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            planes.data_ptr<float>(),
            depth_limits.data_ptr<float>(),
            coverages.data_ptr<float>(),
            inv_softness,
            skip_covered_back,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            out_front.data_ptr<float>(),
            T_front.data_ptr<float>(),
            out_back.data_ptr<float>(),
            T_back.data_ptr<float>(),
            last_ids.data_ptr<int32_t>()
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <uint32_t CDIM>
void launch_rasterize_hybrid_bwd(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::Tensor &planes,
    const at::Tensor &depth_limits,
    const at::Tensor &coverages,
    const float inv_softness,
    const bool skip_covered_back,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    const at::Tensor &T_front,
    const at::Tensor &T_back,
    const at::Tensor &last_ids,
    const at::Tensor &v_out_front,
    const at::Tensor &v_T_front,
    const at::Tensor &v_out_back,
    const at::Tensor &v_T_back,
    at::Tensor &v_means2d,
    at::Tensor &v_conics,
    at::Tensor &v_colors,
    at::Tensor &v_opacities,
    at::Tensor &v_planes,
    at::Tensor &v_depth_limits
) {
    const uint32_t C = tile_offsets.size(0);
    const uint32_t N = means2d.size(1);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width = tile_offsets.size(2);
    const uint32_t n_isects = flatten_ids.size(0);
    if (C == 0 || tile_height == 0 || tile_width == 0 || n_isects == 0) {
        return;
    }

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};
    const int64_t shmem_size = hybrid_bwd_shmem(tile_size, CDIM);
    check_hybrid_launch((const void *)rasterize_hybrid_bwd_kernel<CDIM>, shmem_size);

    rasterize_hybrid_bwd_kernel<CDIM>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            reinterpret_cast<const float2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<const float3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            planes.data_ptr<float>(),
            depth_limits.data_ptr<float>(),
            coverages.data_ptr<float>(),
            inv_softness,
            skip_covered_back,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            T_front.data_ptr<float>(),
            T_back.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_out_front.data_ptr<float>(),
            v_T_front.data_ptr<float>(),
            v_out_back.data_ptr<float>(),
            v_T_back.data_ptr<float>(),
            reinterpret_cast<float2 *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<float3 *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>(),
            v_planes.data_ptr<float>(),
            v_depth_limits.data_ptr<float>()
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define GMESH_HYBRID_CHANNEL_SWITCH(CHANNELS, LAUNCH)                          \
    switch (CHANNELS) {                                                        \
        LAUNCH(1)                                                              \
        LAUNCH(2)                                                              \
        LAUNCH(3)                                                              \
        LAUNCH(4)                                                              \
        LAUNCH(5)                                                              \
        LAUNCH(8)                                                              \
        LAUNCH(9)                                                              \
        LAUNCH(16)                                                             \
        LAUNCH(17)                                                             \
        LAUNCH(32)                                                             \
        LAUNCH(33)                                                             \
    default:                                                                   \
        AT_ERROR("Unsupported number of channels: ", CHANNELS);               \
    }

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_hybrid_fwd(
    const at::Tensor &means2d,      // [C, N, 2]
    const at::Tensor &conics,       // [C, N, 3]
    const at::Tensor &colors,       // [C, N, channels]
    const at::Tensor &opacities,    // [C, N]
    const at::Tensor &planes,       // [C, N, 4]
    const at::Tensor &depth_limits, // [C, H, W]
    const at::Tensor &coverages,    // [C, H, W]
    const double softness,
    const bool skip_covered_back,
    const int64_t image_width,
    const int64_t image_height,
    const int64_t tile_size,
    const at::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    const int64_t C = tile_offsets.size(0);
    const uint32_t channels = colors.size(-1);
    auto opts = means2d.options();

    at::Tensor out_front = at::empty({C, image_height, image_width, channels}, opts);
    at::Tensor out_back = at::empty({C, image_height, image_width, channels}, opts);
    at::Tensor T_front = at::empty({C, image_height, image_width}, opts);
    at::Tensor T_back = at::empty({C, image_height, image_width}, opts);
    at::Tensor last_ids =
        at::empty({C, image_height, image_width, 2}, opts.dtype(at::kInt));

    const float inv_softness = static_cast<float>(1.0 / softness);

#define __LAUNCH__(CDIM)                                                       \
    case CDIM:                                                                 \
        launch_rasterize_hybrid_fwd<CDIM>(                                     \
            means2d, conics, colors, opacities, planes, depth_limits,          \
            coverages, inv_softness, skip_covered_back,                        \
            static_cast<uint32_t>(image_width),                                \
            static_cast<uint32_t>(image_height),                               \
            static_cast<uint32_t>(tile_size), tile_offsets, flatten_ids,       \
            out_front, T_front, out_back, T_back, last_ids                     \
        );                                                                     \
        break;

    GMESH_HYBRID_CHANNEL_SWITCH(channels, __LAUNCH__)
#undef __LAUNCH__

    return std::make_tuple(out_front, T_front, out_back, T_back, last_ids);
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
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    const uint32_t channels = colors.size(-1);

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_conics = at::zeros_like(conics);
    at::Tensor v_colors = at::zeros_like(colors);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_planes = at::zeros_like(planes);
    at::Tensor v_depth_limits = at::zeros_like(depth_limits);

    const float inv_softness = static_cast<float>(1.0 / softness);

#define __LAUNCH__(CDIM)                                                       \
    case CDIM:                                                                 \
        launch_rasterize_hybrid_bwd<CDIM>(                                     \
            means2d, conics, colors, opacities, planes, depth_limits,          \
            coverages, inv_softness, skip_covered_back,                        \
            static_cast<uint32_t>(image_width),                                \
            static_cast<uint32_t>(image_height),                               \
            static_cast<uint32_t>(tile_size), tile_offsets, flatten_ids,       \
            T_front, T_back, last_ids, v_out_front, v_T_front, v_out_back,     \
            v_T_back, v_means2d, v_conics, v_colors, v_opacities, v_planes,    \
            v_depth_limits                                                     \
        );                                                                     \
        break;

    GMESH_HYBRID_CHANNEL_SWITCH(channels, __LAUNCH__)
#undef __LAUNCH__

    return std::make_tuple(
        v_means2d, v_conics, v_colors, v_opacities, v_planes, v_depth_limits
    );
}

} // namespace gmesh
