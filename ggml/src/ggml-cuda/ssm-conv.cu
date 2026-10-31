#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// Fused conv input, new conv state and convolution.
//
// The unfused graph has these nodes:
//   conv_input = concat(conv_state, qkv)            [d_conv - 1 + n_t, nr, n_s]
//   cache      = cpy(last d_conv - 1 columns of conv_input)
//   y          = silu(ssm_conv(conv_input, w))      [nr, n_t, n_s]
// This kernel does not write conv_input. A thread owns one channel of one sequence. It reads the
// columns of conv_input from conv_state and qkv with coalesced loads (the threads of a warp have
// adjacent channels), keeps them in registers and writes y. The block with the last tokens also
// writes the new conv state. Each output is the same sum of products in the same sequence as in
// ssm_conv_f32 and ssm_conv_long_token_f32, so the result does not change.
template <size_t d_conv, int split_n_t>
static __global__ void ssm_conv_concat_silu_f32(
        const float * state, const float * xn, const float * w, float * dst, float * cache,
        const int64_t nr, const int64_t n_t,
        const int64_t xn_nb0, const int64_t xn_nb1, const int64_t xn_nb2,
        const int64_t w_nb1, const int64_t cache_nb1) {
    ggml_cuda_pdl_lc();
    constexpr int n_state = d_conv - 1;

    const int64_t c  = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    const int64_t s  = blockIdx.x;
    const int64_t t0 = (int64_t) blockIdx.z * split_n_t;

    ggml_cuda_pdl_sync();

    const int64_t n_local = min((int64_t) split_n_t, n_t - t0);

    const float * st = state + (s * nr + c) * n_state;
    const float * xc = xn + s * xn_nb2 + c * xn_nb1;

    // column v of the conv input
    auto column = [&](const int64_t v) -> float {
        return v < n_state ? st[v] : xc[(v - n_state) * xn_nb0];
    };

    float wc[d_conv];
#pragma unroll
    for (int j = 0; j < (int) d_conv; j++) {
        wc[j] = w[c * w_nb1 + j];
    }

    float col[n_state + split_n_t];
#pragma unroll
    for (int m = 0; m < n_state + split_n_t; m++) {
        col[m] = m < n_local + n_state ? column(t0 + m) : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < split_n_t; i++) {
        if (i < n_local) {
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < (int) d_conv; j++) {
                sumf += col[i + j] * wc[j];
            }
            // ssm_conv_f32 adds the bias here, 0 without a bias. The add changes the sign of a -0.0 sum.
            sumf += 0.0f;
            dst[(s * n_t + t0 + i) * nr + c] = ggml_cuda_op_silu_single(sumf);
        }
    }

    // the new conv state is the last d_conv - 1 columns of the conv input
    if (t0 + split_n_t >= n_t) {
#pragma unroll
        for (int k = 0; k < n_state; k++) {
            cache[s * cache_nb1 + c * n_state + k] = column(n_t + k);
        }
    }
}

template <size_t d_conv>
static void ssm_conv_concat_silu_f32_cuda(
        const float * state, const float * xn, const float * w, float * dst, float * cache,
        const int64_t nr, const int64_t n_t, const int64_t n_s,
        const int64_t xn_nb0, const int64_t xn_nb1, const int64_t xn_nb2,
        const int64_t w_nb1, const int64_t cache_nb1, cudaStream_t stream) {
    constexpr int threads   = 128;
    constexpr int split_n_t = 32;
    GGML_ASSERT(nr % threads == 0);

    const dim3 blocks(n_s, nr / threads, (n_t + split_n_t - 1) / split_n_t);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, stream);
    ggml_cuda_kernel_launch(ssm_conv_concat_silu_f32<d_conv, split_n_t>, launch_params,
            state, xn, w, dst, cache, nr, n_t, xn_nb0, xn_nb1, xn_nb2, w_nb1, cache_nb1);
}

bool ggml_cuda_ssm_conv_concat_supported(
        const ggml_tensor * concat, const ggml_tensor * cpy, const ggml_tensor * ssm_conv, const ggml_tensor * silu) {
    if (concat->op != GGML_OP_CONCAT || cpy->op != GGML_OP_CPY || ssm_conv->op != GGML_OP_SSM_CONV ||
        silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        return false;
    }

    // concat along the first dimension: [d_conv - 1, nr, n_s] (contiguous) and [n_t, nr, n_s] (any strides)
    const ggml_tensor * state = concat->src[0];
    const ggml_tensor * xn    = concat->src[1];
    if (ggml_get_op_params_i32(concat, 0) != 0 || concat->type != GGML_TYPE_F32 ||
        state->type != GGML_TYPE_F32 || xn->type != GGML_TYPE_F32 || !ggml_is_contiguous(state) ||
        state->ne[3] != 1 || xn->ne[3] != 1 || xn->ne[1] != state->ne[1] || xn->ne[2] != state->ne[2]) {
        return false;
    }

    const int64_t n_state = state->ne[0];
    const int64_t nr      = state->ne[1];
    const int64_t n_s     = state->ne[2];
    const int64_t n_t     = xn->ne[0];
    // the grid is (n_s, nr / 128, n_t / 32), and the limit of the last two dimensions of a grid is 65535
    if (n_state < 1 || n_t < 1 || nr % 128 != 0 || nr / 128 > 65535 || (n_t + 31) / 32 > 65535 ||
        concat->ne[0] != n_state + n_t) {
        return false;
    }

    // the convolution: [d_conv, nr] weights, the output is contiguous [nr, n_t, n_s]
    const ggml_tensor * w = ssm_conv->src[1];
    if (ssm_conv->src[0] != concat || w->type != GGML_TYPE_F32 || w->ne[0] != n_state + 1 || w->ne[1] != nr ||
        w->ne[2] != 1 || w->ne[3] != 1 || w->nb[0] != sizeof(float) ||
        ssm_conv->type != GGML_TYPE_F32 || silu->src[0] != ssm_conv || silu->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(silu) || silu->ne[0] != nr || silu->ne[1] != n_t || silu->ne[2] != n_s ||
        silu->ne[3] != 1) {
        return false;
    }
    switch (n_state + 1) {
        case 3: case 4: case 5: case 7: case 9: case 15: break;
        default: return false;
    }

    // the copy: the last d_conv - 1 columns of the concat into rows of the cache, one row for each sequence
    const ggml_tensor * view  = cpy->src[0];
    const ggml_tensor * cache = cpy->src[1];
    if (view->op != GGML_OP_VIEW || view->view_src != concat || view->type != GGML_TYPE_F32 ||
        view->ne[0] != n_state || view->ne[1] != nr || view->ne[2] != n_s || view->ne[3] != 1 ||
        view->nb[0] != sizeof(float) || view->nb[1] != concat->nb[1] || view->nb[2] != concat->nb[2] ||
        view->view_offs != (size_t) n_t * sizeof(float)) {
        return false;
    }
    if (cache->op != GGML_OP_VIEW || cache->type != GGML_TYPE_F32 || cpy->type != GGML_TYPE_F32 ||
        cache->ne[0] != n_state * nr || cache->ne[1] != n_s || cache->ne[2] != 1 || cache->ne[3] != 1 ||
        cache->nb[0] != sizeof(float) || cache->nb[1] % sizeof(float) != 0) {
        return false;
    }

    // the strides of the columns of the new input are in floats
    return xn->nb[0] % sizeof(float) == 0 && xn->nb[1] % sizeof(float) == 0 && xn->nb[2] % sizeof(float) == 0;
}

void ggml_cuda_op_ssm_conv_concat(
        ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * cpy,
        const ggml_tensor * ssm_conv, ggml_tensor * silu) {
    GGML_ASSERT(ggml_cuda_ssm_conv_concat_supported(concat, cpy, ssm_conv, silu));

    const ggml_tensor * state = concat->src[0];
    const ggml_tensor * xn    = concat->src[1];
    const ggml_tensor * w     = ssm_conv->src[1];
    const ggml_tensor * cache = cpy->src[1];

    const int64_t nr  = state->ne[1];
    const int64_t n_t = xn->ne[0];
    const int64_t n_s = state->ne[2];

    const float * state_d = (const float *) state->data;
    const float * xn_d    = (const float *) xn->data;
    const float * w_d     = (const float *) w->data;
    float *       dst_d   = (float *) silu->data;
    float *       cache_d = (float *) cache->data;

    const int64_t xn_nb0 = xn->nb[0] / sizeof(float);
    const int64_t xn_nb1 = xn->nb[1] / sizeof(float);
    const int64_t xn_nb2 = xn->nb[2] / sizeof(float);
    const int64_t w_nb1  = w->nb[1] / sizeof(float);
    const int64_t cache_nb1 = cache->nb[1] / sizeof(float);

    cudaStream_t stream = ctx.stream();

    switch (w->ne[0]) {
#define GGML_CUDA_SSM_CONV_CONCAT_CASE(NC) \
        case NC: ssm_conv_concat_silu_f32_cuda<NC>(state_d, xn_d, w_d, dst_d, cache_d, nr, n_t, n_s, \
                    xn_nb0, xn_nb1, xn_nb2, w_nb1, cache_nb1, stream); break;
        GGML_CUDA_SSM_CONV_CONCAT_CASE(3)
        GGML_CUDA_SSM_CONV_CONCAT_CASE(4)
        GGML_CUDA_SSM_CONV_CONCAT_CASE(5)
        GGML_CUDA_SSM_CONV_CONCAT_CASE(7)
        GGML_CUDA_SSM_CONV_CONCAT_CASE(9)
        GGML_CUDA_SSM_CONV_CONCAT_CASE(15)
#undef GGML_CUDA_SSM_CONV_CONCAT_CASE
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}
