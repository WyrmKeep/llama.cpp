#pragma once

// DMK timing-only ablation switches for the CUDA backend (private build, not meant for upstream).
//
//   DMK_ABLATE=mmq,fattn,...   skip the GPU work of the listed classes. Everything else is unchanged: the graph, the
//                              fusion decisions, the buffers, the CUDA graph capture and the launches of the other
//                              classes. The answers are garbage. Use it only to bound what a class costs.
//   DMK_ABLATE_SYNC=1          direct launches (no CUDA graph) with a stream sync after every ggml node and a table
//                              of wall time per node kind (prefix "DMK_OPS"): at exit, after one second without a node,
//                              and every DMK_ABLATE_SYNC_DUMP graphs.
//   DMK_ABLATE_SYNC_DUMP=N     also print and reset that table every N graph computes.
//   DMK_TIMING=1               host timing lines on stderr (prefix "DMK_TIMING"), see also src/llama-context.cpp.
//
// All of it is read once in ggml_cuda_init(). With none of the variables set every switch is one load and one branch.
// The checks are in host code only: no device code is touched.

#include "ggml.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

enum dmk_class : uint32_t {
    DMK_MMQ         = 1u << 0,  // mul_mat_q launches (matmul and stream-k fixup, also the fused gate/up/glu one), quantize kept
    DMK_MMVQ        = 1u << 1,  // mul_mat_vec_q launches, quantize kept
    DMK_MMVF        = 1u << 2,  // mul_mat_vec_f launches
    DMK_MMF         = 1u << 3,  // mul_mat_f launches
    DMK_CUBLAS      = 1u << 4,  // cuBLAS fallback matmuls
    DMK_QUANT_MMQ   = 1u << 5,  // quantize_mmq_q8_1 (activation quantization in front of MMQ)
    DMK_QUANT_MMVQ  = 1u << 6,  // quantize_q8_1 (activation quantization in front of MMVQ)
    DMK_FATTN       = 1u << 7,  // FLASH_ATTN_EXT, the whole op
    DMK_GDN         = 1u << 8,  // GATED_DELTA_NET, the whole op (also the variant that writes the state snapshots)
    DMK_GLU         = 1u << 9,  // standalone GLU ops and the fused unary+mul (silu(z)*x, sigmoid(g)*x)
    DMK_NORM        = 1u << 10, // RMS_NORM and NORM, with the fused rms_norm+mul(+add / +rope / +scale) kernels
    DMK_L2NORM      = 1u << 11, // L2_NORM
    DMK_CPY         = 1u << 12, // CPY, DUP, CONT, SET
    DMK_SET_ROWS    = 1u << 13, // SET_ROWS (KV cache write)
    DMK_CONCAT      = 1u << 14, // CONCAT
    DMK_GET_ROWS    = 1u << 15, // GET_ROWS (embedding lookup, recurrent state gather, output row selection)
    DMK_SSM_CONV    = 1u << 16, // SSM_CONV, also the fused conv+add+silu
    DMK_ELEMENTWISE = 1u << 17, // ADD, SUB, MUL, DIV, SCALE, CLAMP, SQR, ..., REPEAT, UNARY, and their fused kernels
    DMK_ROPE        = 1u << 18, // ROPE (a fused rms_norm+mul+rope kernel counts for NORM and ROPE)
    DMK_HEAD        = 1u << 19, // the whole output-head MUL_MAT (the one with >= DMK_HEAD_MIN_ROWS rows), quantize included
    DMK_MMID        = 1u << 20, // MUL_MAT_ID, the whole op (not in this model)
    DMK_OTHER       = 1u << 21, // every other ggml op (SOFT_MAX, SUM_ROWS, CUMSUM, TRI, SOLVE_TRI, ...)
};

struct dmk_item {
    const char * name;
    uint32_t     bits;
};

#define DMK_ALL_BITS ((1u << 22) - 1)
#define DMK_COPIES   (DMK_CPY | DMK_SET_ROWS | DMK_CONCAT | DMK_GET_ROWS)
#define DMK_QUANTIZE (DMK_QUANT_MMQ | DMK_QUANT_MMVQ)
#define DMK_MATMUL   (DMK_MMQ | DMK_MMVQ | DMK_MMVF | DMK_MMF | DMK_CUBLAS | DMK_MMID)

inline constexpr dmk_item dmk_items[] = {
    { "mmq",         DMK_MMQ },
    { "mmvq",        DMK_MMVQ },
    { "mmvf",        DMK_MMVF },
    { "mmf",         DMK_MMF },
    { "cublas",      DMK_CUBLAS },
    { "quantize",    DMK_QUANTIZE },
    { "quantize_mmq",  DMK_QUANT_MMQ },
    { "quantize_mmvq", DMK_QUANT_MMVQ },
    { "fattn",       DMK_FATTN },
    { "gdn",         DMK_GDN },
    { "glu",         DMK_GLU },
    { "norm",        DMK_NORM },
    { "l2norm",      DMK_L2NORM },
    { "cpy",         DMK_CPY },
    { "set_rows",    DMK_SET_ROWS },
    { "concat",      DMK_CONCAT },
    { "get_rows",    DMK_GET_ROWS },
    { "copies",      DMK_COPIES },
    { "ssm_conv",    DMK_SSM_CONV },
    { "elementwise", DMK_ELEMENTWISE },
    { "rope",        DMK_ROPE },
    { "head",        DMK_HEAD },
    { "mmid",        DMK_MMID },
    { "other",       DMK_OTHER },
    { "matmul",      DMK_MATMUL },      // every matmul kernel, activation quantization kept
    { "all",         DMK_ALL_BITS },
};

// output-head matmul: no other weight of the model has this many rows (token_embd.weight has 248,320)
#define DMK_HEAD_MIN_ROWS 65536

inline uint32_t g_dmk_mask   = 0;     // classes to skip, 0 = unchanged behaviour
inline bool     g_dmk_sync   = false; // DMK_ABLATE_SYNC
inline int      g_dmk_dump   = 0;     // DMK_ABLATE_SYNC_DUMP
inline bool     g_dmk_timing = false; // DMK_TIMING

static inline bool dmk_skip(uint32_t bits) {
    return (g_dmk_mask & bits) != 0;
}

// skip the statement(s) when one of the classes is ablated; the caller keeps its control flow (return values, fusion counts)
#define DMK_RUN(bits, ...) do { if (!dmk_skip(bits)) { __VA_ARGS__; } } while (0)

static inline int64_t dmk_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline bool dmk_is_head(const ggml_tensor * src0) {
    return src0->ne[1] >= DMK_HEAD_MIN_ROWS && src0->ne[2] == 1 && src0->ne[3] == 1;
}

// class of a ggml node that is skipped as a whole by ggml_cuda_compute_forward.
// 0 = never skipped here: MUL_MAT, whose kernels are gated one by one (mmq.cu, mmvq.cu, mmvf.cu, mmf.cu, ggml-cuda.cu)
static inline uint32_t dmk_node_class(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_MUL_MAT:
            return 0;
        case GGML_OP_MUL_MAT_ID:
            return DMK_MMID;
        case GGML_OP_FLASH_ATTN_EXT:
            return DMK_FATTN;
        case GGML_OP_GATED_DELTA_NET:
            return DMK_GDN;
        case GGML_OP_GLU:
            return DMK_GLU;
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
            return DMK_NORM;
        case GGML_OP_L2_NORM:
            return DMK_L2NORM;
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_SET:
            return DMK_CPY;
        case GGML_OP_SET_ROWS:
            return DMK_SET_ROWS;
        case GGML_OP_CONCAT:
            return DMK_CONCAT;
        case GGML_OP_GET_ROWS:
            return DMK_GET_ROWS;
        case GGML_OP_SSM_CONV:
            return DMK_SSM_CONV;
        case GGML_OP_ROPE:
            return DMK_ROPE;
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_ADD_ID:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ACC:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
        case GGML_OP_CLAMP:
        case GGML_OP_REPEAT:
        case GGML_OP_UNARY:
            return DMK_ELEMENTWISE;
        default:
            return DMK_OTHER;
    }
}

// parse DMK_ABLATE (comma or blank separated), DMK_ABLATE_SYNC, DMK_ABLATE_SYNC_DUMP and DMK_TIMING; called once from ggml_cuda_init()
inline void dmk_ablate_init() {
    const char * env_timing = getenv("DMK_TIMING");
    g_dmk_timing = env_timing != nullptr && atoi(env_timing) != 0;

    const char * env_sync = getenv("DMK_ABLATE_SYNC");
    g_dmk_sync = env_sync != nullptr && atoi(env_sync) != 0;

    const char * env_dump = getenv("DMK_ABLATE_SYNC_DUMP");
    g_dmk_dump = env_dump != nullptr ? atoi(env_dump) : 0;

    const char * env = getenv("DMK_ABLATE");
    uint32_t mask = 0;
    if (env != nullptr) {
        const size_t n = strlen(env);
        size_t pos = 0;
        while (pos < n) {
            while (pos < n && (env[pos] == ',' || env[pos] == ' ' || env[pos] == '\t')) {
                pos++;
            }
            size_t end = pos;
            while (end < n && env[end] != ',' && env[end] != ' ' && env[end] != '\t') {
                end++;
            }
            if (end > pos) {
                bool found = false;
                for (const dmk_item & item : dmk_items) {
                    if (strlen(item.name) == end - pos && strncmp(item.name, env + pos, end - pos) == 0) {
                        mask |= item.bits;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    // a typo must not turn into a silent no-op ablation
                    fprintf(stderr, "DMK_ABLATE: unknown item '%.*s' in \"%s\"\n", (int) (end - pos), env + pos, env);
                    abort();
                }
            }
            pos = end;
        }
    }
    g_dmk_mask = mask;

    if (mask != 0 || g_dmk_sync || g_dmk_timing) {
        fprintf(stderr, "DMK_ABLATE active: mask=0x%x items=\"%s\" sync=%d sync_dump=%d timing=%d\n",
                mask, env != nullptr ? env : "", (int) g_dmk_sync, g_dmk_dump, (int) g_dmk_timing);
    }
}
