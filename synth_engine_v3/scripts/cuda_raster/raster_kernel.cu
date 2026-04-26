// CUDA polygon rasterizer — per-pixel point-in-polygon (Phase OPT-2.2, doc/20).
//
// Replaces PIL/freetype CPU mask raster with batched on-device rasterization.
// Each thread (b, y, x) computes whether pixel center (x+0.5, y+0.5) is
// inside the b-th glyph's polygon by counting horizontal-ray crossings against
// its edges (even/odd fill rule).
//
// Lab-application notes (doc/20 §2):
//   * Coalesced edge access — Lab 8 (SpMV / JDS pattern). Edges are packed
//     contiguously per glyph; the per-batch offsets array partitions them.
//     We load each edge once and broadcast against all pixels of one glyph.
//   * Stencil sweep — Lab 4. Each thread is one output pixel, similar to
//     a stencil but with a polygon predicate instead of a fixed kernel.
//   * No reduction needed — the cross-count is summed inline.
//
// Future optimization (V2): tile-based shared-memory edge cache, scan-line
// active-edge prefix (Lab 7). Skipped here for V1 simplicity.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(EXPR)                                                       \
    do {                                                                       \
        cudaError_t _e = (EXPR);                                               \
        TORCH_CHECK(_e == cudaSuccess, "CUDA error: ",                         \
                    cudaGetErrorString(_e));                                   \
    } while (0)

// Thread block: 16×16 pixels.
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;

// One thread per output pixel of one glyph.
// Grid = (ceil(W / BLOCK_X), ceil(H / BLOCK_Y), B)
__global__ void raster_kernel(
    const float4* __restrict__ edges,          // (E_total, 4) — (x0,y0,x1,y1)
    const int* __restrict__ glyph_off,         // (B+1,) prefix sums into edges
    float* __restrict__ out_mask,              // (B, 1, H, W) float in [0,1]
    int B, int H, int W
) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= W || y >= H || b >= B) return;

    const int e0 = glyph_off[b];
    const int e1 = glyph_off[b + 1];
    const float px = static_cast<float>(x) + 0.5f;
    const float py = static_cast<float>(y) + 0.5f;

    int crossings = 0;
    for (int e = e0; e < e1; ++e) {
        const float4 ed = edges[e];
        const float x0 = ed.x, y0 = ed.y, x1 = ed.z, y1 = ed.w;
        // Horizontal ray going right at y = py. Edge crosses ray iff
        // (y0 <= py) != (y1 <= py).
        const bool a = (y0 <= py);
        const bool c = (y1 <= py);
        if (a == c) continue;
        // x-coord of intersection
        const float t = (py - y0) / (y1 - y0);
        const float xi = x0 + t * (x1 - x0);
        if (xi <= px) {
            crossings += 1;
        }
    }
    const float val = (crossings & 1) ? 1.0f : 0.0f;
    out_mask[((b * 1 + 0) * H + y) * W + x] = val;
}


torch::Tensor raster_forward(
    torch::Tensor edges,            // (E_total, 4) float32 (CUDA)
    torch::Tensor glyph_off,        // (B+1,) int32 (CUDA)
    int H, int W
) {
    TORCH_CHECK(edges.is_cuda(), "edges must be a CUDA tensor");
    TORCH_CHECK(glyph_off.is_cuda(), "glyph_off must be a CUDA tensor");
    TORCH_CHECK(edges.dtype() == torch::kFloat32, "edges must be float32");
    TORCH_CHECK(glyph_off.dtype() == torch::kInt32, "glyph_off must be int32");
    TORCH_CHECK(edges.dim() == 2 && edges.size(1) == 4, "edges must be (E, 4)");
    TORCH_CHECK(glyph_off.dim() == 1, "glyph_off must be 1-D");

    const int B = glyph_off.size(0) - 1;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(edges.device());
    auto out = torch::zeros({B, 1, H, W}, opts);

    if (B == 0) return out;

    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);

    raster_kernel<<<grid, block>>>(
        reinterpret_cast<const float4*>(edges.data_ptr<float>()),
        glyph_off.data_ptr<int>(),
        out.data_ptr<float>(),
        B, H, W
    );
    CUDA_CHECK(cudaGetLastError());
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("raster_forward", &raster_forward,
          "Batched polygon rasterizer (per-pixel point-in-polygon, even-odd fill)",
          py::arg("edges"), py::arg("glyph_off"),
          py::arg("H"), py::arg("W"));
}
