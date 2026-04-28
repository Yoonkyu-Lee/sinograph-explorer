// Fused photometric augment kernel — Phase OPT-3 (doc/20).
//
// Replaces 4 separate kornia/torch ops in the augment chain with one
// single CUDA pass:
//
//     brightness * canvas
//     + (canvas - per_sample_mean) * contrast + per_sample_mean
//     + noise_seed * noise_std        // pre-generated normal samples
//     invert ? (1 - canvas) : canvas
//     clamp(0, 1)
//
// Design notes:
//   * Lab 1 — one thread per output element baseline.
//   * **Register tiling / N-coarsening** — each thread processes 4 pixels
//     along the channel-row dimension, reusing the per-sample brightness /
//     contrast / mean / noise_std / invert flag held in registers across
//     the 4 outputs. Cuts register pressure / launch count.
//   * **Memory traffic** — single read + single write of canvas instead
//     of 4 read+write passes, ~4× memory-bandwidth saving for the
//     elementwise photometric chain.
//   * Per-sample mean is a single torch.mean reduction kept on the Python
//     side (kornia / torch use cuDNN reductions which are already
//     near-optimal for this).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(EXPR)                                                       \
    do {                                                                       \
        cudaError_t _e = (EXPR);                                               \
        TORCH_CHECK(_e == cudaSuccess, "CUDA error: ",                         \
                    cudaGetErrorString(_e));                                   \
    } while (0)

// 4-pixel register tiling along the inner dim.
constexpr int COARSE_X = 4;
constexpr int BLOCK_X = 32;     // → 32 threads × 4 pixels = 128 cols/block
constexpr int BLOCK_Y = 8;


// canvas shape: (N, 3, H, W) float [0, 1]
// All per-sample buffers are float32[N], invert_flag is uint8[N] (0/1).
// noise_seed is the pre-generated standard-normal tensor (N, 3, H, W) — we
// scale it by per-sample noise_std inside the kernel, so caller doesn't
// have to do a separate multiply pass.
__global__ void photometric_fused_kernel(
    float* __restrict__ canvas,
    const float* __restrict__ noise_seed,
    const float* __restrict__ brightness,    // (N,)
    const float* __restrict__ contrast,      // (N,)
    const float* __restrict__ mean_per_sample,  // (N,) — pre-computed
    const float* __restrict__ noise_std,     // (N,) values already in [0,1] pixel scale
    const uint8_t* __restrict__ invert_flag, // (N,)
    int N, int C, int H, int W
) {
    const int n = blockIdx.z / C;
    const int c = blockIdx.z % C;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int x0 = (blockIdx.x * BLOCK_X + threadIdx.x) * COARSE_X;
    if (n >= N || y >= H || x0 >= W) return;

    // Per-sample params loaded once into registers, reused for COARSE_X
    // pixels — Lab 4 register-tiling pattern.
    const float br = brightness[n];
    const float ct = contrast[n];
    const float mu = mean_per_sample[n];
    const float ns = noise_std[n];
    const float inv = invert_flag[n] ? 1.0f : 0.0f;

    const int base = ((n * C + c) * H + y) * W + x0;

    #pragma unroll
    for (int dx = 0; dx < COARSE_X; ++dx) {
        const int x = x0 + dx;
        if (x >= W) break;
        const int idx = base + dx;
        float p = canvas[idx];

        // contrast: (p - mean) * ct + mean
        p = (p - mu) * ct + mu;
        // brightness
        p *= br;
        // gaussian noise
        if (ns > 0.0f) {
            p += noise_seed[idx] * ns;
        }
        // invert (sample-conditional)
        // invert == 1 → p ← 1 - p ; invert == 0 → no-op
        // expressed branchlessly: p = inv * (1 - p) + (1 - inv) * p
        p = fmaf(inv, 1.0f - 2.0f * p, p);
        // clamp [0, 1]
        p = fminf(fmaxf(p, 0.0f), 1.0f);
        canvas[idx] = p;
    }
}


void photometric_fused(
    torch::Tensor canvas,            // (N, 3, H, W) float32 CUDA, modified in place
    torch::Tensor noise_seed,        // (N, 3, H, W) standard-normal samples
    torch::Tensor brightness,        // (N,) float32
    torch::Tensor contrast,          // (N,) float32
    torch::Tensor mean_per_sample,   // (N,) float32
    torch::Tensor noise_std,         // (N,) float32 (already pixel-scaled)
    torch::Tensor invert_flag        // (N,) uint8
) {
    TORCH_CHECK(canvas.is_cuda() && canvas.dtype() == torch::kFloat32);
    TORCH_CHECK(canvas.dim() == 4);
    const int N = canvas.size(0);
    const int C = canvas.size(1);
    const int H = canvas.size(2);
    const int W = canvas.size(3);

    TORCH_CHECK(noise_seed.is_cuda() && noise_seed.dtype() == torch::kFloat32);
    TORCH_CHECK(noise_seed.numel() == canvas.numel());
    TORCH_CHECK(brightness.is_cuda() && brightness.size(0) == N);
    TORCH_CHECK(contrast.is_cuda() && contrast.size(0) == N);
    TORCH_CHECK(mean_per_sample.is_cuda() && mean_per_sample.size(0) == N);
    TORCH_CHECK(noise_std.is_cuda() && noise_std.size(0) == N);
    TORCH_CHECK(invert_flag.is_cuda() && invert_flag.size(0) == N);
    TORCH_CHECK(invert_flag.dtype() == torch::kUInt8);

    if (N == 0) return;

    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid(((W + COARSE_X - 1) / COARSE_X + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              N * C);

    photometric_fused_kernel<<<grid, block>>>(
        canvas.data_ptr<float>(),
        noise_seed.data_ptr<float>(),
        brightness.data_ptr<float>(),
        contrast.data_ptr<float>(),
        mean_per_sample.data_ptr<float>(),
        noise_std.data_ptr<float>(),
        invert_flag.data_ptr<uint8_t>(),
        N, C, H, W
    );
    CUDA_CHECK(cudaGetLastError());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("photometric_fused", &photometric_fused,
          "Fused brightness + contrast + invert + gaussian_noise (Lab 1 + register tiling)",
          py::arg("canvas"), py::arg("noise_seed"),
          py::arg("brightness"), py::arg("contrast"),
          py::arg("mean_per_sample"), py::arg("noise_std"),
          py::arg("invert_flag"));
}
