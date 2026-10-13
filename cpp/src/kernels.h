#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace seg {

constexpr int kMaxClasses = 8;  // seg_infer.h SEG_MAX_CLASSES 와 동일

cudaError_t launch_normalize(const uint8_t* d_in, float* d_out, size_t n, float mean, float std,
                             cudaStream_t stream);

cudaError_t launch_argmax_count(const float* d_logits, uint8_t* d_mask, int* d_counts, int batch,
                                int classes, int hw, cudaStream_t stream);

}  // namespace seg
