// ubp-convert.cpp — Convert a GGUF model to UBP sparse format using the BED algorithm.
//
// Usage:
//   llama-ubp-convert -i model.gguf -o model.ubp.gguf [-N 4] [-s 0.8]
//
// For each eligible linear-layer weight tensor (F32 or F16, 2-D, excluding lm_head):
//   1. Run the BED (Block Expansion and Division) algorithm to select blocks.
//   2. Apply WROS rotation to the selected blocks.
//   3. Store the result as a GGML_TYPE_UBP_F16 tensor in the output GGUF.
//
// Tensors that are not converted are copied verbatim.

#include "ggml.h"
#include "ggml-ubp.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool ends_with(const std::string & s, const std::string & suf) {
    if (s.size() < suf.size()) return false;
    return s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// Names of tensors to skip (kept dense).
static bool should_skip(const std::string & name) {
    // lm_head / output projection — keep dense
    if (ends_with(name, "output.weight"))  return true;
    if (ends_with(name, "lm_head.weight")) return true;
    // Embedding tables, norms, bias tensors: skip 1-D and 3-D+
    return false;
}

// ---------------------------------------------------------------------------
// BED: Block Expansion and Division
//
// References: Algorithm 1 & 2 in the paper.
//
// Input : row-major weight matrix W[n_rows][n_cols] (as flat F32 vector)
//         block size N, number of blocks to keep = m_l
// Output: ridx — starting row index of each selected block (sorted)
//         cidx — column index of each selected block
//         Both have m_l entries.
// ---------------------------------------------------------------------------

// Importance score of block starting at (row, col) with size N.
static float block_score(const float * W, int n_rows, int n_cols, int row, int col, int N) {
    if (row + N > n_rows) return -std::numeric_limits<float>::infinity();
    float s = 0.f;
    for (int n = 0; n < N; n++) {
        s += std::fabs(W[(row + n) * n_cols + col]);
    }
    return s;
}

// BED algorithm for ONE column (1×N pruning along output-channel dimension).
// Returns the row indices of selected blocks for this column.
static std::vector<int32_t> bed_column(
        const float * col_data, // W[:, j], length n_rows
        int           n_rows,
        int           N,
        int           m_col)    // number of blocks to select for this column
{
    // Available block start positions and their scores.
    // We maintain a "score set" S and corresponding index set R (both shrink as we pick).
    // For efficiency we use a flat vector of (score, original_row_idx) and re-sort each round.

    struct Entry { float score; int orig_row; };
    std::vector<Entry> S;
    S.reserve(n_rows);
    for (int i = 0; i + N <= n_rows; i++) {
        float sc = 0.f;
        for (int n = 0; n < N; n++) sc += std::fabs(col_data[i + n]);
        S.push_back({sc, i});
    }

    std::vector<int32_t> selected;
    selected.reserve(m_col);

    // Scratch: quick-sorted scores each round.
    // BED complexity: O(b_l * m_l) — for each of m_col picks we scan+update S.
    for (int pick = 0; pick < m_col && !S.empty(); pick++) {
        // Find highest-scoring block.
        int best_idx = 0;
        for (int i = 1; i < (int)S.size(); i++) {
            if (S[i].score > S[best_idx].score) best_idx = i;
        }
        Entry best = S[best_idx];
        selected.push_back(best.orig_row);

        // BED: update scores of blocks that would overlap (become adjacent to chosen block).
        // For blocks starting at rows [best.orig_row - N + 1 .. best.orig_row - 1]:
        //   new_score = score_assuming_chosen_block_removed + delta
        //   = S[k] + S[best.orig_row + N] - S[best.orig_row]  (simplified as per Algorithm 1 line 8)
        // Then remove entries that overlap with chosen block from S (rows best.orig_row .. best.orig_row+N-1).

        // Apply score recalculation to adjacent entries (those that overlap with chosen block)
        for (auto & e : S) {
            int r = e.orig_row;
            // Overlap with chosen block: r in [best.orig_row - N + 1, best.orig_row + N - 1]
            if (r >= best.orig_row - N + 1 && r < best.orig_row) {
                // This block partially overlaps; BED recalculates its score assuming
                // the chosen elements don't belong to it.
                // Recompute score of the elements NOT in [best.orig_row .. best.orig_row+N-1].
                // Elements of block r that are outside chosen block: rows r .. best.orig_row-1
                // After the chosen block is removed, the block's elements shift — but BED instead
                // looks for an expansion that captures adjacent elements.  The score update from
                // Algorithm 1 line 8: S[k-n] += S[k-n+N] - S[k]  for n in 1..N-1.
                // For a column-wise view, the "expansion" adds elements from beyond the chosen block.
                int beyond_row = best.orig_row + N + (best.orig_row - 1 - r);
                float beyond = (beyond_row < n_rows) ?
                    std::fabs(col_data[beyond_row]) : 0.f;
                float replaced = std::fabs(col_data[best.orig_row + (best.orig_row - 1 - r)]);
                // Simplified: adjust score by swapping out the overlapping element
                e.score += beyond - replaced;
            }
        }

        // Remove entries that include chosen block's rows (overlap).
        S.erase(std::remove_if(S.begin(), S.end(), [&](const Entry & e) {
            int r = e.orig_row;
            // Block [r, r+N) overlaps [best.orig_row, best.orig_row+N)?
            return (r < best.orig_row + N) && (r + N > best.orig_row);
        }), S.end());
    }

    return selected;
}

// ---------------------------------------------------------------------------
// Build UBP blob from a dense F32 matrix W[n_rows][n_cols].
// ---------------------------------------------------------------------------
static std::vector<uint8_t> build_ubp_blob(
        const float * W,
        int32_t       n_rows,
        int32_t       n_cols,
        int           N,
        float         sparsity) // fraction of weights to PRUNE (0..1)
{
    // Total blocks per column and number to keep.
    const int blocks_per_col = std::max(0, n_rows - N + 1);
    const int total_blocks   = blocks_per_col * n_cols;
    // Number of blocks to KEEP across all columns.
    const int m_total = std::max(1, (int)std::round(total_blocks * (1.f - sparsity) / N));
    // Keep roughly equal blocks per column.
    const int m_per_col = std::max(1, m_total / n_cols);

    // Run BED per column.
    std::vector<std::vector<int32_t>> col_blocks(n_cols);
    uint32_t nnz = 0;
    for (int j = 0; j < n_cols; j++) {
        // Extract column j.
        std::vector<float> col(n_rows);
        for (int i = 0; i < n_rows; i++) col[i] = W[i * n_cols + j];
        col_blocks[j] = bed_column(col.data(), n_rows, N, m_per_col);
        nnz += (uint32_t)col_blocks[j].size();
    }

    // Build CSC arrays.
    std::vector<ggml_fp16_t> data_f16(nnz * N);
    std::vector<int32_t>     ridx(nnz);
    std::vector<int32_t>     cptr(n_cols + 1);

    uint32_t ptr = 0;
    for (int j = 0; j < n_cols; j++) {
        cptr[j] = (int32_t)ptr;
        for (int32_t row_i : col_blocks[j]) {
            ridx[ptr] = row_i;
            // WROS rotation: shift data left by (row_i % N)
            const int rot = row_i % N;
            for (int n = 0; n < N; n++) {
                float val = ((row_i + n) < n_rows) ? W[(row_i + n) * n_cols + j] : 0.f;
                int dst_pos = (n + rot) % N;  // rotated position
                data_f16[ptr * N + dst_pos] = ggml_fp32_to_fp16(val);
            }
            ptr++;
        }
    }
    cptr[n_cols] = (int32_t)ptr;
    assert(ptr == nnz);

    // Pack into blob.
    const size_t blob_sz = ubp_blob_size(N, n_cols, nnz);
    std::vector<uint8_t> blob(blob_sz);

    ubp_header_t * hdr = reinterpret_cast<ubp_header_t *>(blob.data());
    hdr->magic   = UBP_MAGIC;
    hdr->version = UBP_VERSION;
    hdr->N       = (uint32_t)N;
    hdr->n_rows  = (uint32_t)n_rows;
    hdr->n_cols  = (uint32_t)n_cols;
    hdr->nnz     = nnz;

    ggml_fp16_t * dst_data = reinterpret_cast<ggml_fp16_t *>(blob.data() + sizeof(ubp_header_t));
    int32_t  * dst_ridx = reinterpret_cast<int32_t *>(dst_data + nnz * N);
    int32_t  * dst_cptr = dst_ridx + nnz;

    memcpy(dst_data, data_f16.data(), nnz * N * sizeof(ggml_fp16_t));
    memcpy(dst_ridx, ridx.data(),    nnz     * sizeof(int32_t));
    memcpy(dst_cptr, cptr.data(),    (n_cols + 1) * sizeof(int32_t));

    const float actual_sparsity = 1.f - (float)(nnz * N) / (float)(n_rows * n_cols);
    printf("  nnz=%u blocks, effective sparsity=%.1f%%, blob=%zu KB\n",
           nnz, actual_sparsity * 100.f, blob_sz / 1024);

    return blob;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s -i INPUT.gguf -o OUTPUT.gguf [-N BLOCK_SIZE] [-s SPARSITY]\n"
        "  -N  block size (default: 4, i.e. 1×4 pattern)\n"
        "  -s  target pruning sparsity 0..1 (default: 0.8)\n",
        prog);
}

int main(int argc, char ** argv) {
    const char * input_path  = nullptr;
    const char * output_path = nullptr;
    int          N           = 4;
    float        sparsity    = 0.8f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) { input_path  = argv[++i]; }
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) { output_path = argv[++i]; }
        else if (!strcmp(argv[i], "-N") && i + 1 < argc) { N           = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) { sparsity    = (float)atof(argv[++i]); }
        else { print_usage(argv[0]); return 1; }
    }
    if (!input_path || !output_path) { print_usage(argv[0]); return 1; }
    if (N < 1 || N > 64) { fprintf(stderr, "Invalid block size N=%d\n", N); return 1; }
    if (sparsity < 0.f || sparsity >= 1.f) { fprintf(stderr, "Sparsity must be in [0,1)\n"); return 1; }

    printf("Loading %s ...\n", input_path);

    // --- Load source GGUF ---
    struct ggml_context * src_ctx = nullptr;
    struct gguf_init_params iparams = { .no_alloc = false, .ctx = &src_ctx };
    struct gguf_context * src_gguf  = gguf_init_from_file(input_path, iparams);
    if (!src_gguf) { fprintf(stderr, "Failed to load %s\n", input_path); return 1; }

    const int n_tensors = gguf_get_n_tensors(src_gguf);
    printf("Loaded %d tensors.\n", n_tensors);

    // --- Create output GGUF ---
    struct gguf_context * dst_gguf = gguf_init_empty();

    // Copy all KV metadata.
    gguf_set_kv(dst_gguf, src_gguf);

    // --- Process tensors ---
    struct ggml_init_params gparams = {
        .mem_size   = (size_t)4 * 1024 * 1024 * 1024ULL, // 4 GB scratch
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    struct ggml_context * dst_ctx = ggml_init(gparams);

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(src_gguf, i);
        struct ggml_tensor * t = ggml_get_tensor(src_ctx, name);
        if (!t) { fprintf(stderr, "Tensor %s not found\n", name); continue; }

        const bool is_2d = (t->ne[2] == 1 && t->ne[3] == 1);
        const bool skip  = should_skip(name);
        const bool eligible = is_2d && !skip
                           && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16)
                           && t->ne[0] >= (int64_t)N && t->ne[1] >= (int64_t)N;

        if (eligible) {
            const int64_t n_in  = t->ne[0]; // input channels (columns of W)
            const int64_t n_out = t->ne[1]; // output channels (rows of W)
            printf("[%d/%d] Converting %-60s [%ld×%ld] → UBP_F16 N=%d s=%.0f%%\n",
                   i + 1, n_tensors, name, (long)n_out, (long)n_in, N, sparsity * 100.f);

            // Convert to F32 if needed.
            std::vector<float> W_f32(n_out * n_in);
            if (t->type == GGML_TYPE_F32) {
                memcpy(W_f32.data(), t->data, W_f32.size() * sizeof(float));
            } else { // F16
                const ggml_fp16_t * src16 = (const ggml_fp16_t *)t->data;
                for (size_t k = 0; k < W_f32.size(); k++) {
                    W_f32[k] = ggml_fp16_to_fp32(src16[k]);
                }
            }

            std::vector<uint8_t> blob = build_ubp_blob(
                W_f32.data(), (int32_t)n_out, (int32_t)n_in, N, sparsity);

            // Create a 1-D UBP_F16 tensor to hold the blob.
            const int64_t blob_ne = (int64_t)blob.size();
            struct ggml_tensor * ubp_t = ggml_new_tensor_1d(dst_ctx, GGML_TYPE_UBP_F16, blob_ne);
            ggml_set_name(ubp_t, name);
            memcpy(ubp_t->data, blob.data(), blob.size());
            gguf_add_tensor(dst_gguf, ubp_t);

            // Store n_out in KV so the graph builder can recover the logical shape.
            const std::string meta_key = std::string("ubp.n_out.") + name;
            gguf_set_val_u32(dst_gguf, meta_key.c_str(), (uint32_t)n_out);
        } else {
            printf("[%d/%d] Copying   %-60s [type=%d]\n", i + 1, n_tensors, name, (int)t->type);
            // Add tensor verbatim (data already in src_ctx memory).
            gguf_add_tensor(dst_gguf, t);
        }
    }

    // --- Write output ---
    printf("Writing %s ...\n", output_path);
    gguf_write_to_file(dst_gguf, output_path, false);
    printf("Done.\n");

    ggml_free(dst_ctx);
    gguf_free(dst_gguf);
    gguf_free(src_gguf);
    return 0;
}
