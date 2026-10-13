// GPU 전/후처리 커널
//  - normalize_u8: uint8 gray -> float32 정규화 (Python common.normalize 와 동일 식)
//  - argmax_count: logits[N,C,H,W] -> 클래스 마스크 + patch별 클래스 면적 집계
//    (logit 전체를 호스트로 복사하지 않고 GPU에서 축약해 D2H 전송량을 1/(4*C)로 줄임)
#include "kernels.h"

#include <cuda_runtime.h>

namespace seg {

__global__ void normalize_u8_kernel(const uint8_t* __restrict__ in, float* __restrict__ out,
                                    size_t n, float scale, float bias) {
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        out[i] = static_cast<float>(in[i]) * scale + bias;
    }
}

// grid = (blocks_per_patch, batch). 각 블록은 shared memory에 면적을 모은 뒤 전역 메모리에 1회 atomicAdd.
__global__ void argmax_count_kernel(const float* __restrict__ logits, uint8_t* __restrict__ mask,
                                    int* __restrict__ counts, int classes, int hw) {
    __shared__ int s_counts[kMaxClasses];
    const int b = blockIdx.y;
    if (threadIdx.x < kMaxClasses) s_counts[threadIdx.x] = 0;
    __syncthreads();

    const float* base = logits + static_cast<size_t>(b) * classes * hw;
    uint8_t* mbase = mask + static_cast<size_t>(b) * hw;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hw; i += gridDim.x * blockDim.x) {
        float best = base[i];
        int arg = 0;
        for (int c = 1; c < classes; ++c) {
            float v = base[static_cast<size_t>(c) * hw + i];
            if (v > best) { best = v; arg = c; }
        }
        mbase[i] = static_cast<uint8_t>(arg);
        atomicAdd(&s_counts[arg], 1);
    }
    __syncthreads();
    if (threadIdx.x < classes && s_counts[threadIdx.x] > 0) {
        atomicAdd(&counts[b * classes + threadIdx.x], s_counts[threadIdx.x]);
    }
}

cudaError_t launch_normalize(const uint8_t* d_in, float* d_out, size_t n, float mean, float std,
                             cudaStream_t stream) {
    const float scale = 1.0f / (255.0f * std);
    const float bias = -mean / std;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads < 4096 ? (n + threads - 1) / threads : 4096);
    normalize_u8_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n, scale, bias);
    return cudaGetLastError();
}

cudaError_t launch_argmax_count(const float* d_logits, uint8_t* d_mask, int* d_counts, int batch,
                                int classes, int hw, cudaStream_t stream) {
    cudaError_t err = cudaMemsetAsync(d_counts, 0, sizeof(int) * batch * classes, stream);
    if (err != cudaSuccess) return err;
    const int threads = 256;
    const int blocks_per_patch = 64;
    dim3 grid(blocks_per_patch, batch);
    argmax_count_kernel<<<grid, threads, 0, stream>>>(d_logits, d_mask, d_counts, classes, hw);
    return cudaGetLastError();
}

}  // namespace seg
