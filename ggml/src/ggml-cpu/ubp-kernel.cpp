// ubp-kernel.cpp — UBP SpMV forward kernel
//
// Implements y = W_ubp * x  using Compressed Sparse Column (CSC) format.
//
// WROS (Weight Rotating and Output Stationary) note:
//   Weights are pre-rotated offline.  For block starting at row r, stored as:
//     stored[k] = original[(k + r%N) % N]
//   During inference the output registers remain stationary; only one register
//   is written to memory per output tile.  The current implementation fully
//   accumulates each block into the output buffer for correctness; the full
//   output-stationary optimisation (avoiding the load/store) is a future TODO.

#include "ubp-kernel.h"
#include "ggml-impl.h"

#include <cstring>  // memset
#include <cstdint>

#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

// ---------------------------------------------------------------------------
// Scalar fallback: N-generic SpMV
// ---------------------------------------------------------------------------
static void ubp_spmv_scalar(
        const ubp_header_t * hdr,
        const uint16_t     * data,  // F16 [nnz*N]
        const int32_t      * ridx,  // [nnz]
        const int32_t      * cptr,  // [n_cols+1]
        const float        * x,
        float              * y)
{
    const int N      = (int)hdr->N;
    const int n_cols = (int)hdr->n_cols;

    for (int j = 0; j < n_cols; j++) {
        const float x_j = x[j];
        for (int k = cptr[j]; k < cptr[j + 1]; k++) {
            const int          row_i = ridx[k];
            const uint16_t   * blk   = data + (size_t)k * N;
            float            * out   = y + row_i;
            for (int n = 0; n < N; n++) {
                out[n] += GGML_FP16_TO_FP32(blk[n]) * x_j;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEON path for N == 4 (most common block size, matches NEON float lane width)
// ---------------------------------------------------------------------------
#if defined(__ARM_NEON)
static void ubp_spmv_neon_n4(
        const uint32_t   n_cols,
        const uint32_t   nnz,
        const uint16_t * data,  // F16 [nnz*4]
        const int32_t  * ridx,  // [nnz]
        const int32_t  * cptr,  // [n_cols+1]
        const float    * x,
        float          * y)
{
    for (uint32_t j = 0; j < n_cols; j++) {
        const float32x4_t vx = vdupq_n_f32(x[j]);
        for (int32_t k = cptr[j]; k < cptr[j + 1]; k++) {
            const int32_t row_i = ridx[k];
            // Load 4 × F16, convert to F32
            const float16x4_t vw16 = vld1_f16(
                reinterpret_cast<const __fp16 *>(data + (size_t)k * 4));
            const float32x4_t vw32 = vcvt_f32_f16(vw16);
            // y[row_i..row_i+3] += vw32 * x_j  (unaligned load/store OK on AArch64)
            float32x4_t vy = vld1q_f32(y + row_i);
            vy = vmlaq_f32(vy, vw32, vx);
            vst1q_f32(y + row_i, vy);
        }
    }
    (void)nnz;
}
#endif // __ARM_NEON

// ---------------------------------------------------------------------------
// Public entry: dispatch to best available path
// ---------------------------------------------------------------------------
void ubp_spmv_f32(const void * blob, const float * x, float * y)
{
    const ubp_header_t * hdr  = ubp_header(blob);
    const uint16_t     * data = ubp_data(blob);
    const int32_t      * ridx = ubp_ridx(blob, hdr->N, hdr->nnz);
    const int32_t      * cptr = ubp_cptr(blob, hdr->N, hdr->nnz);

#if defined(__ARM_NEON)
    if (hdr->N == 4) {
        ubp_spmv_neon_n4(hdr->n_cols, hdr->nnz, data, ridx, cptr, x, y);
        return;
    }
#endif
    ubp_spmv_scalar(hdr, data, ridx, cptr, x, y);
}

// ---------------------------------------------------------------------------
// Full forward pass with batch + threading
// ---------------------------------------------------------------------------
void ubp_compute_forward(
        const struct ggml_compute_params * params,
        struct ggml_tensor               * dst,
        const struct ggml_tensor         * W,
        const struct ggml_tensor         * x_src)
{
    const ubp_header_t * hdr   = ubp_header(W->data);
    const int64_t        n_out = (int64_t)hdr->n_rows;
    // x_src shape: [n_in, batch1, batch2, batch3]
    const int64_t ne11 = x_src->ne[1];
    const int64_t ne12 = x_src->ne[2];
    const int64_t ne13 = x_src->ne[3];
    const int64_t n_batch = ne11 * ne12 * ne13;

    const int ith = params->ith;
    const int nth = params->nth;

    // Partition batch rows among threads
    for (int64_t ib = ith; ib < n_batch; ib += nth) {
        const int64_t i3 = ib / (ne12 * ne11);
        const int64_t i2 = (ib - i3 * ne12 * ne11) / ne11;
        const int64_t i1 = ib - i3 * ne12 * ne11 - i2 * ne11;

        const float * x_ptr = (const float *)((const char *)x_src->data
                              + i1 * x_src->nb[1]
                              + i2 * x_src->nb[2]
                              + i3 * x_src->nb[3]);

        float * y_ptr = (float *)((char *)dst->data
                        + i1 * dst->nb[1]
                        + i2 * dst->nb[2]
                        + i3 * dst->nb[3]);

        // Zero the output slice
        memset(y_ptr, 0, (size_t)n_out * sizeof(float));
        ubp_spmv_f32(W->data, x_ptr, y_ptr);
    }
}
