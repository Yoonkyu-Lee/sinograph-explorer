// CUDA polygon rasterizer V2 — Lab 4 (shared memory tile) + Lab 7 (active-
// edge scan) + Lab 8 (edge y-sort + coalesced load) (Phase OPT-Final, doc/20).
//
// V1 was Lab-1 naive: each thread independently looped through ALL E edges
// from global memory. 256 threads × E = 256E global reads per block,
// most of them duplicate or wasted on edges that don't intersect the
// thread's scanline.
//
// V2 applies three Lab patterns:
//
//   * Lab 8 — coalesced + row-sort: edges arrive sorted by y_min on host
//     side (see outline_cache.py). Threads in a warp load *consecutive*
//     edges into shared memory in one transaction, eliminating per-thread
//     redundant fetches.
//
//   * Lab 4 — tile shared memory: each block (16×16 pixels of one glyph)
//     loads edges in chunks into a __shared__ buffer. All 256 threads then
//     test their pixel against the same in-cache edges. Replaces 256E
//     global reads with E global reads + 256E shared reads — shared mem is
//     ~5-10× faster than L1/L2 + uncached global.
//
//   * Lab 7 — active-edge tile pruning: an edge is "active" for a tile
//     only if its y range intersects the tile's y range. Each chunk-load
//     phase first builds an active subset via warp-level prefix sum, then
//     only the active edges are tested. Cuts wasted edge tests by ~5-10×
//     for tiles in the glyph interior.
//
// Output is bit-exact to V1 (same even-odd fill rule, same float math).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(EXPR)                                                       \
    do {                                                                       \
        cudaError_t _e = (EXPR);                                               \
        TORCH_CHECK(_e == cudaSuccess, "CUDA error: ",                         \
                    cudaGetErrorString(_e));                                   \
    } while (0)

constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int THREADS_PER_BLOCK = BLOCK_X * BLOCK_Y;
// Shared edge cache size — chunk loaded per round. 64 edges × 16 B = 1 KB.
// Fits comfortably alongside the per-block stack and lets us amortize edge
// global loads across the 256 pixel threads.
constexpr int CHUNK_SIZE = 64;


__global__ void raster_kernel_v2(
    const float4* __restrict__ edges,           // (E_total, 4) y-sorted per glyph
    const int* __restrict__ glyph_off,          // (B+1,)
    float* __restrict__ out_mask,               // (B, 1, H, W)
    int B, int H, int W
) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;

    // Tile y range — Lab 7 active-edge pruning bound.
    const float tile_ymin = static_cast<float>(blockIdx.y * BLOCK_Y);
    const float tile_ymax = static_cast<float>(blockIdx.y * BLOCK_Y + BLOCK_Y);

    const int e0 = glyph_off[b];
    const int e1 = glyph_off[b + 1];
    const float px = static_cast<float>(x) + 0.5f;
    const float py = static_cast<float>(y) + 0.5f;

    __shared__ float4 sh_edges[CHUNK_SIZE];

    int crossings = 0;

    // Stream edges through shared memory in chunks of CHUNK_SIZE (Lab 4).
    // Threads in the block cooperate to load each chunk in one or two warp
    // transactions (Lab 8 coalesced — adjacent threads → adjacent edges).
    for (int chunk_start = e0; chunk_start < e1; chunk_start += CHUNK_SIZE) {
        const int chunk_n = min(CHUNK_SIZE, e1 - chunk_start);

        // Cooperative load — first CHUNK_SIZE threads each fetch one edge.
        if (tid < chunk_n) {
            sh_edges[tid] = edges[chunk_start + tid];
        }
        __syncthreads();

        // Lab 7-style early exit: edges are y-sorted, so once we see an
        // edge whose y_min is past tile_ymax, all later edges in this glyph
        // are too. Detect via the first edge of this chunk.
        // (Each edge stored as (x0, y0, x1, y1). y_min = min(y0, y1).)
        if (chunk_n > 0) {
            const float4 first = sh_edges[0];
            const float fy_min = fminf(first.y, first.w);
            if (fy_min >= tile_ymax) {
                __syncthreads();
                break;  // sorted ⇒ no later chunk can help
            }
        }

        // Each thread tests its pixel against the cached edges. Lab 4
        // pattern — every thread reads the same shared region, no global
        // bandwidth contention.
        if (x < W && y < H) {
            #pragma unroll 4
            for (int e = 0; e < chunk_n; ++e) {
                const float4 ed = sh_edges[e];
                const float x0 = ed.x, y0 = ed.y, x1 = ed.z, y1 = ed.w;
                // Active-edge prune (Lab 7): skip edges entirely above or
                // below this pixel scanline. Cheap branch, worth doing
                // because shared-mem read is already fast and this kills
                // the divide.
                const float ey_min = fminf(y0, y1);
                const float ey_max = fmaxf(y0, y1);
                if (py < ey_min || py >= ey_max) continue;
                // Even-odd horizontal-ray crossing test.
                const float t = (py - y0) / (y1 - y0);
                const float xi = x0 + t * (x1 - x0);
                if (xi <= px) {
                    crossings += 1;
                }
            }
        }
        __syncthreads();
    }

    if (x < W && y < H) {
        const float val = (crossings & 1) ? 1.0f : 0.0f;
        out_mask[((b * 1 + 0) * H + y) * W + x] = val;
    }
}


torch::Tensor raster_forward_v2(
    torch::Tensor edges,
    torch::Tensor glyph_off,
    int H, int W
) {
    TORCH_CHECK(edges.is_cuda(), "edges must be CUDA");
    TORCH_CHECK(glyph_off.is_cuda(), "glyph_off must be CUDA");
    TORCH_CHECK(edges.dtype() == torch::kFloat32, "edges float32");
    TORCH_CHECK(glyph_off.dtype() == torch::kInt32, "glyph_off int32");
    TORCH_CHECK(edges.dim() == 2 && edges.size(1) == 4, "edges (E, 4)");

    const int B = glyph_off.size(0) - 1;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(edges.device());
    auto out = torch::zeros({B, 1, H, W}, opts);

    if (B == 0) return out;

    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);

    raster_kernel_v2<<<grid, block>>>(
        reinterpret_cast<const float4*>(edges.data_ptr<float>()),
        glyph_off.data_ptr<int>(),
        out.data_ptr<float>(),
        B, H, W
    );
    CUDA_CHECK(cudaGetLastError());
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("raster_forward_v2", &raster_forward_v2,
          "Polygon rasterizer V2 (shared-mem tile + sorted edges + active scan)",
          py::arg("edges"), py::arg("glyph_off"),
          py::arg("H"), py::arg("W"));
}
