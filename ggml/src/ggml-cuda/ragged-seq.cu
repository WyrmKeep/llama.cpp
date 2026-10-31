#include "ragged-seq.cuh"
#include "unary.cuh"

#include <cstdint>
#include <type_traits>

// ---------------------------------------------------------------------------------------------
// SSM_CONV_RAGGED
//
// One thread owns one channel. The conv input of a sequence is its old conv state followed by its
// new inputs, and the output at token t is sum_j input[t + j] * w[j]. The sum runs in the same
// order as in ssm_conv_f32 (ssm-conv.cu), so the result is the same as for one sequence per batch row.
// A CTA works on at most ssm_conv_ragged_chunk tokens of one sequence; the CTAs that are past the
// end of their sequence leave at once (the grid is sized for the worst case, n_tokens, because the
// lengths are only known on the device).
// ---------------------------------------------------------------------------------------------

constexpr int ssm_conv_ragged_chunk   = 32;
constexpr int ssm_conv_ragged_threads = 128;

template <bool apply_silu, int d_conv>
static __global__ void __launch_bounds__(ssm_conv_ragged_threads)
ssm_conv_ragged_f32(const float * states, const float * x, const float * w, const int32_t * seq_off,
                    float * dst, float * state_out, const int64_t d_inner) {
    ggml_cuda_pdl_lc();

    const int seq = blockIdx.z;
    const int c   = blockIdx.y * blockDim.x + threadIdx.x;

    ggml_cuda_pdl_sync();

    const int t0      = seq_off[seq];
    const int n_t     = seq_off[seq + 1] - t0;
    const int t_start = blockIdx.x * ssm_conv_ragged_chunk;

    // the CTA that holds the last token of the sequence writes the new conv state (CTA 0 for an empty sequence)
    const bool writes_state = n_t == 0 ? blockIdx.x == 0 : (t_start < n_t && t_start + ssm_conv_ragged_chunk >= n_t);
    if ((t_start >= n_t && !writes_state) || c >= d_inner) {
        return;
    }

    const int n_in_chunk = max(0, min(ssm_conv_ragged_chunk, n_t - t_start));

    const float * st = states + (int64_t) seq * (d_conv - 1) * d_inner + (int64_t) c * (d_conv - 1);

    float wr[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; j++) {
        wr[j] = w[(int64_t) c * d_conv + j];
    }

    // conv input positions t_start .. t_start + d_conv - 2: the old state first, then the new inputs
    float prev[d_conv - 1];
#pragma unroll
    for (int j = 0; j < d_conv - 1; j++) {
        const int u = t_start + j;
        prev[j] = u < d_conv - 1 ? st[u] : x[(int64_t) (t0 + u - (d_conv - 1)) * d_inner + c];
    }

    float xv[ssm_conv_ragged_chunk];
#pragma unroll
    for (int i = 0; i < ssm_conv_ragged_chunk; i++) {
        xv[i] = i < n_in_chunk ? x[(int64_t) (t0 + t_start + i) * d_inner + c] : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < ssm_conv_ragged_chunk; i++) {
        if (i < n_in_chunk) {
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < d_conv - 1; j++) {
                sumf += prev[j] * wr[j];
            }
            sumf += xv[i] * wr[d_conv - 1];
            dst[(int64_t) (t0 + t_start + i) * d_inner + c] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;

#pragma unroll
            for (int j = 0; j < d_conv - 2; j++) {
                prev[j] = prev[j + 1];
            }
            prev[d_conv - 2] = xv[i];
        }
    }

    if (writes_state) {
        float * so = state_out + (int64_t) seq * (d_conv - 1) * d_inner + (int64_t) c * (d_conv - 1);
#pragma unroll
        for (int j = 0; j < d_conv - 1; j++) {
            so[j] = prev[j];
        }
    }
}

template <bool apply_silu>
static void ssm_conv_ragged_f32_cuda(
        const float * states, const float * x, const float * w, const int32_t * seq_off, float * dst,
        const int64_t d_conv, const int64_t d_inner, const int64_t n_tokens, const int64_t n_seqs, cudaStream_t stream) {
    GGML_ASSERT(d_inner % ssm_conv_ragged_threads == 0);

    float * state_out = dst + n_tokens * d_inner;

    const int64_t n_chunks = std::max<int64_t>(1, (n_tokens + ssm_conv_ragged_chunk - 1) / ssm_conv_ragged_chunk);
    const dim3 blocks(n_chunks, d_inner / ssm_conv_ragged_threads, n_seqs);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        const ggml_cuda_kernel_launch_params launch_params(blocks, ssm_conv_ragged_threads, 0, stream);
        ggml_cuda_kernel_launch(ssm_conv_ragged_f32<apply_silu, kNC>, launch_params,
                                states, x, w, seq_off, dst, state_out, d_inner);
    };

    switch (d_conv) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}

bool ggml_cuda_ssm_conv_ragged_supported(const ggml_tensor * op) {
    const int64_t d_conv  = op->src[2]->ne[0];
    const int64_t d_inner = op->src[2]->ne[1];
    return d_inner % ssm_conv_ragged_threads == 0 &&
           (d_conv == 3 || d_conv == 4 || d_conv == 5 || d_conv == 7 || d_conv == 9 || d_conv == 15) &&
           op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
           op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_I32;
}

void ggml_cuda_op_ssm_conv_ragged(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * states  = dst->src[0]; // {d_conv - 1, d_inner, n_seqs}
    const ggml_tensor * x       = dst->src[1]; // {d_inner, n_tokens}
    const ggml_tensor * w       = dst->src[2]; // {d_conv, d_inner}
    const ggml_tensor * seq_off = dst->src[3]; // {n_seqs + 1}

    const int64_t d_conv   = w->ne[0];
    const int64_t d_inner  = w->ne[1];
    const int64_t n_tokens = x->ne[1];
    const int64_t n_seqs   = states->ne[2];

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_tokens + (d_conv - 1) * n_seqs);
    GGML_ASSERT(ggml_is_contiguous(states));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const bool silu = ggml_get_op_params_i32(dst, 0) != 0;

    if (silu) {
        ssm_conv_ragged_f32_cuda<true>((const float *) states->data, (const float *) x->data, (const float *) w->data,
                                       (const int32_t *) seq_off->data, (float *) dst->data,
                                       d_conv, d_inner, n_tokens, n_seqs, ctx.stream());
    } else {
        ssm_conv_ragged_f32_cuda<false>((const float *) states->data, (const float *) x->data, (const float *) w->data,
                                        (const int32_t *) seq_off->data, (float *) dst->data,
                                        d_conv, d_inner, n_tokens, n_seqs, ctx.stream());
    }
}

// ---------------------------------------------------------------------------------------------
// GATED_DELTA_NET_RAGGED
//
// The per-token recurrence of gated_delta_net_cuda<S_v, false, false> (gated_delta_net.cu): the same
// arithmetic in the same order, so the result is bit-identical to the one-token-at-a-time path.
// The tokens of all sequences are in one row; the sequence of a CTA gives its first token and its
// token count through seq_off. Never uses the chunked (tensor core) path.
// ---------------------------------------------------------------------------------------------

constexpr int gdn_ragged_cols_per_warp = 4;

template <int S_v, int cols_per_warp = gdn_ragged_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_ragged_cuda(const float *   q,
                            const float *   k,
                            const float *   v,
                            const float *   g,
                            const float *   beta,
                            const float *   curr_state,
                            const int32_t * seq_off,
                            float *         dst,
                            float *         state,
                            int64_t         H,
                            int64_t         sq1,
                            int64_t         sq2,
                            int64_t         sv1,
                            int64_t         sv2,
                            int64_t         sb1,
                            int64_t         sb2,
                            const uint3     neqk1_magic,
                            float           scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp * lanes_per_col;
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;

    float s_shard[rows_per_lane];

    ggml_cuda_pdl_sync();

    const int64_t t0       = seq_off[sequence];
    const int64_t n_tokens = seq_off[sequence + 1] - t0;

    float * attn_data = dst + (t0 * H + h_idx) * S_v;

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int64_t t = 0; t < n_tokens; t++) {
        const float * q_t = q + (t0 + t) * sq2 + iq1 * sq1;
        const float * k_t = k + (t0 + t) * sq2 + iq1 * sq1;
        const float * v_t = v + (t0 + t) * sv2 + h_idx * sv1;

        const int64_t gb_offset = (t0 + t) * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset;

        const float beta_val = *beta_t;

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        const float g_val = expf(*g_t);

        // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

        // delta[col] = (v[col] - g * kv[col]) * beta
        float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

        // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
        // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }

        float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

        if (lane_in_col == 0) {
            attn_data[col] = attn_col * scale;
        }

        attn_data += S_v * H;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * lanes_per_col + lane_in_col;
        state[col * S_v + i] = s_shard[r];
    }
}

bool ggml_cuda_gated_delta_net_ragged_supported(const ggml_tensor * op) {
    const int64_t S_v = op->src[2]->ne[0];
    for (int i = 0; i < 6; i++) {
        if (op->src[i]->type != GGML_TYPE_F32) {
            return false;
        }
    }
    return (S_v == 16 || S_v == 32 || S_v == 64 || S_v == 128) && op->src[0]->ne[0] == S_v &&
           op->src[3]->ne[0] == 1 && op->src[6]->type == GGML_TYPE_I32 && op->type == GGML_TYPE_F32;
}

void ggml_cuda_op_gated_delta_net_ragged(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_cache) {
    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];
    const ggml_tensor * seq_off   = dst->src[6];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_state->ne[3];

    GGML_ASSERT(src_q->ne[1] == src_k->ne[1]);
    const int64_t neqk1 = src_q->ne[1];

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats; g and beta share the strides of beta
    const int64_t sq1 = src_q->nb[1] / sizeof(float);
    const int64_t sq2 = src_q->nb[2] / sizeof(float);
    const int64_t sv1 = src_v->nb[1] / sizeof(float);
    const int64_t sv2 = src_v->nb[2] / sizeof(float);
    const int64_t sb1 = src_beta->nb[1] / sizeof(float);
    const int64_t sb2 = src_beta->nb[2] / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    float * dst_d   = (float *) dst->data;
    float * state_d = state_cache != nullptr ? state_cache : dst_d + S_v * H * n_tokens;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int nsm       = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    // same CTA shape rule as launch_gated_delta_net (the arithmetic does not depend on it)
    int num_warps = 4;
    while (num_warps > 1 && H * n_seqs * (S_v / (gdn_ragged_cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }

    const dim3 grid_dims(H, n_seqs, (S_v + gdn_ragged_cols_per_warp * num_warps - 1) / (gdn_ragged_cols_per_warp * num_warps));
    const dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);

    const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, ctx.stream());

    const float *   q_d       = (const float *) src_q->data;
    const float *   k_d       = (const float *) src_k->data;
    const float *   v_d       = (const float *) src_v->data;
    const float *   g_d       = (const float *) src_g->data;
    const float *   b_d       = (const float *) src_beta->data;
    const float *   s_d       = (const float *) src_state->data;
    const int32_t * seq_off_d = (const int32_t *) seq_off->data;

    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_ragged_cuda<16>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, seq_off_d, dst_d, state_d, H, sq1, sq2, sv1, sv2, sb1, sb2, neqk1_magic, scale);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_ragged_cuda<32>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, seq_off_d, dst_d, state_d, H, sq1, sq2, sv1, sv2, sb1, sb2, neqk1_magic, scale);
            break;
        case 64:
            ggml_cuda_kernel_launch(gated_delta_net_ragged_cuda<64>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, seq_off_d, dst_d, state_d, H, sq1, sq2, sv1, sv2, sb1, sb2, neqk1_magic, scale);
            break;
        case 128:
            ggml_cuda_kernel_launch(gated_delta_net_ragged_cuda<128>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, seq_off_d, dst_d, state_d, H, sq1, sq2, sv1, sv2, sb1, sb2, neqk1_magic, scale);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
