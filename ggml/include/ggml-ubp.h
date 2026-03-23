#pragma once

// ggml-ubp.h — Unaligned Block-wise Pruning (UBP) format and utilities
//
// A GGML_TYPE_UBP_F16 tensor is a 1-D byte blob with the following layout:
//
//   [ubp_header_t header]
//   [ggml_fp16_t  data  [nnz * N]  ]  — F16 non-zero values, stored column-major
//   [int32_t      ridx  [nnz]      ]  — starting ROW of each block (CSC row indices)
//   [int32_t      cptr  [n_cols+1] ]  — CSC column pointers
//
// Storage is Compressed Sparse Column (CSC) format grouped by block:
//   For column j, the non-zero blocks are at rows ridx[cptr[j]..cptr[j+1]-1].
//   Each block contains N consecutive F16 values: W[row, col], W[row+1, col], ..., W[row+N-1, col].
//
// WROS pre-rotation (applied offline during model conversion):
//   For block starting at row r, the values are cyclically rotated left by (r % N):
//   stored[k] = original[(k + r%N) % N]
//   This allows the inference kernel to avoid output-register shuffling between tiles.
//
// The tensor's ne[0] = total byte size of the blob.

#include "ggml.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UBP_MAGIC    UINT32_C(0x55425030) // 'UBP0'
#define UBP_VERSION  UINT32_C(1)

// Header at the start of every UBP blob.
typedef struct {
    uint32_t magic;    // UBP_MAGIC
    uint32_t version;  // UBP_VERSION
    uint32_t N;        // block size (number of consecutive output channels per block)
    uint32_t n_rows;   // number of output channels (n_out)
    uint32_t n_cols;   // number of input channels (n_in)
    uint32_t nnz;      // number of selected (non-zero) blocks
} ubp_header_t;

// Byte size of a UBP blob given header fields.
static inline size_t ubp_blob_size(uint32_t N, uint32_t n_cols, uint32_t nnz) {
    return sizeof(ubp_header_t)
         + (size_t)nnz * N * sizeof(uint16_t)  // F16 data
         + (size_t)nnz * sizeof(int32_t)        // ridx
         + (size_t)(n_cols + 1) * sizeof(int32_t); // cptr
}

// Accessors into a blob pointer (blob = tensor->data).
static inline const ubp_header_t * ubp_header(const void * blob) {
    return (const ubp_header_t *)blob;
}
static inline const uint16_t * ubp_data(const void * blob) {
    return (const uint16_t *)((const uint8_t *)blob + sizeof(ubp_header_t));
}
static inline const int32_t * ubp_ridx(const void * blob, uint32_t N, uint32_t nnz) {
    return (const int32_t *)(ubp_data(blob) + (size_t)nnz * N);
}
static inline const int32_t * ubp_cptr(const void * blob, uint32_t N, uint32_t nnz) {
    return ubp_ridx(blob, N, nnz) + nnz;
}

#ifdef __cplusplus
}
#endif
