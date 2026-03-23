#pragma once

// ubp-kernel.h — UBP SpMV kernel declarations (CPU, with optional NEON path)

#include "ggml-ubp.h"
#include "ggml-cpu-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compute  y += W_ubp * x  for a single batch entry.
// blob   : pointer to the UBP blob (ubp_header_t followed by data/ridx/cptr)
// x      : F32 input vector  [n_in]
// y      : F32 output vector [n_out], must be zeroed by caller
void ubp_spmv_f32(const void * blob, const float * x, float * y);

// Full forward pass: handles batch dimension and threading.
// params : ggml compute params (ith/nth for thread id/count)
// dst    : result tensor (F32 [n_out, batch, ...])
// W      : UBP weight blob tensor (GGML_TYPE_UBP_F16)
// x_src  : input tensor (F32 [n_in, batch, ...])
void ubp_compute_forward(
    const struct ggml_compute_params * params,
    struct ggml_tensor               * dst,
    const struct ggml_tensor         * W,
    const struct ggml_tensor         * x_src);

#ifdef __cplusplus
}
#endif
