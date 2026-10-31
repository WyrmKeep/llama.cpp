#include "kq-mask.cuh"
#include "convert.cuh"

#define CUDA_KQ_MASK_BLOCK_SIZE 128
#define CUDA_KQ_MASK_ROWS       4

// the mask of a token against a cell, see ggml_kq_mask
static __device__ __forceinline__ bool kq_mask_keep(
        const int32_t p0, const uint32_t seqs, const int32_t p1, const uint32_t seq_bit, const bool causal) {
    return p0 >= 0 && (seqs & seq_bit) != 0 && !(causal && p0 > p1);
}

// info rows: pos, seq, x, y. The columns [0, n_kv) are the cells and the columns [n_kv, n_kv + n_tokens) are the tokens.
// Each thread makes nv cells (16 bytes of the mask) of CUDA_KQ_MASK_ROWS tokens, and reads its cells once.
template <typename T, bool vec>
static __global__ void kq_mask_kernel(
        const int32_t * info, T * dst,
        const int64_t n_kv, const int64_t n_tokens, const int64_t si1, const int64_t sd1,
        const bool causal, const bool is_2d) {
    constexpr int nv = vec ? sizeof(int4)/sizeof(T) : 1;

    ggml_cuda_pdl_lc();

    const int64_t c0 = ((int64_t) blockIdx.x*blockDim.x + threadIdx.x)*nv;

    ggml_cuda_pdl_sync();

    if (c0 >= n_kv) {
        return;
    }

    const int32_t * tokens = info + n_kv;

    alignas(sizeof(int4)) int32_t  pos[nv];
    alignas(sizeof(int4)) uint32_t seqs[nv];

    if constexpr (vec) {
#pragma unroll
        for (int k = 0; k < nv; k += 4) {
            *(int4 *) (pos  + k) = *(const int4 *) (info       + c0 + k);
            *(int4 *) (seqs + k) = *(const int4 *) (info + si1 + c0 + k);
        }
    } else {
        pos[0]  = info[c0];
        seqs[0] = info[si1 + c0];
    }

    for (int64_t i0 = (int64_t) blockIdx.y*CUDA_KQ_MASK_ROWS; i0 < n_tokens; i0 += (int64_t) gridDim.y*CUDA_KQ_MASK_ROWS) {
#pragma unroll
        for (int r = 0; r < CUDA_KQ_MASK_ROWS; ++r) {
            const int64_t i = i0 + r;
            if (i >= n_tokens) {
                break;
            }

            const int32_t  p1      = tokens[i];
            const uint32_t seq_bit = 1u << (tokens[si1 + i] & 31);

            alignas(sizeof(int4)) T out[nv];
#pragma unroll
            for (int k = 0; k < nv; ++k) {
                bool keep = kq_mask_keep(pos[k], seqs[k], p1, seq_bit, causal);

                // cells at the position of the token are ordered by the 2D positions of M-RoPE
                if (keep && causal && is_2d && pos[k] == p1) {
                    const int32_t cx = info[2*si1 + c0 + k];
                    const int32_t cy = info[3*si1 + c0 + k];
                    const int32_t tx = tokens[2*si1 + i];
                    const int32_t ty = tokens[3*si1 + i];

                    keep = !(cy > ty || (cy == ty && cx > tx));
                }

                out[k] = keep ? ggml_cuda_cast<T>(0.0f) : ggml_cuda_cast<T>(-INFINITY);
            }

            if constexpr (vec) {
                *(int4 *) (dst + i*sd1 + c0) = *(const int4 *) out;
            } else {
                dst[i*sd1 + c0] = out[0];
            }
        }
    }
}

template <typename T>
static void kq_mask_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * info = dst->src[0];

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];

    GGML_ASSERT(dst->nb[0] == sizeof(T));
    GGML_ASSERT(info->nb[0] == sizeof(int32_t));
    GGML_ASSERT(dst->nb[1] % sizeof(T) == 0 && info->nb[1] % sizeof(int32_t) == 0);

    const bool causal = ggml_get_op_params_i32(dst, 0) != 0;
    const bool is_2d  = ggml_get_op_params_i32(dst, 1) != 0;

    // 16-byte accesses need every row of info and of dst aligned
    const bool vec = n_kv % (sizeof(int4)/sizeof(T)) == 0 &&
        dst->nb[1]  % sizeof(int4) == 0 && (uintptr_t) dst->data  % sizeof(int4) == 0 &&
        info->nb[1] % sizeof(int4) == 0 && (uintptr_t) info->data % sizeof(int4) == 0;

    const int64_t nv    = vec ? sizeof(int4)/sizeof(T) : 1;
    const int64_t n_thr = (n_kv + nv - 1)/nv;

    const dim3 block_dims(CUDA_KQ_MASK_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims(
        (n_thr + CUDA_KQ_MASK_BLOCK_SIZE - 1)/CUDA_KQ_MASK_BLOCK_SIZE,
        std::min<int64_t>((n_tokens + CUDA_KQ_MASK_ROWS - 1)/CUDA_KQ_MASK_ROWS, 65535), 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    const int64_t si1 = info->nb[1]/sizeof(int32_t);
    const int64_t sd1 = dst->nb[1]/sizeof(T);

    if (vec) {
        ggml_cuda_kernel_launch(kq_mask_kernel<T, true>, launch_params,
                (const int32_t *) info->data, (T *) dst->data, n_kv, n_tokens, si1, sd1, causal, is_2d);
    } else {
        ggml_cuda_kernel_launch(kq_mask_kernel<T, false>, launch_params,
                (const int32_t *) info->data, (T *) dst->data, n_kv, n_tokens, si1, sd1, causal, is_2d);
    }
}

void ggml_cuda_op_kq_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->type) {
        case GGML_TYPE_F16:
            kq_mask_cuda<half>(ctx, dst);
            break;
        case GGML_TYPE_F32:
            kq_mask_cuda<float>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
