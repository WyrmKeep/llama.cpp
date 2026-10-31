#include "common.cuh"
#include "ggml.h"

// SSM_CONV_RAGGED and GATED_DELTA_NET_RAGGED: the recurrent ops for one batch of sequences with
// different token counts (see ggml_ssm_conv_ragged and ggml_gated_delta_net_ragged in ggml.h).

bool ggml_cuda_ssm_conv_ragged_supported(const ggml_tensor * op);
bool ggml_cuda_gated_delta_net_ragged_supported(const ggml_tensor * op);

void ggml_cuda_op_ssm_conv_ragged(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// state_cache: when not null, the kernel writes the final states there (row s at state_cache + s * D)
// instead of after the attention scores in dst, so the cpy into the cache can be skipped
void ggml_cuda_op_gated_delta_net_ragged(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_cache);
