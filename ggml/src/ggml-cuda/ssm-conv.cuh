#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// Whether the nodes concat -> cpy (the new conv state into the cache) -> ssm_conv -> silu have the shapes,
// types and strides that ggml_cuda_op_ssm_conv_concat handles. It uses no data pointers.
bool ggml_cuda_ssm_conv_concat_supported(
        const ggml_tensor * concat, const ggml_tensor * cpy, const ggml_tensor * ssm_conv, const ggml_tensor * silu);

// Runs the four nodes as one kernel. It does not write the output of the concat and of the ssm_conv.
void ggml_cuda_op_ssm_conv_concat(
        ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * cpy,
        const ggml_tensor * ssm_conv, ggml_tensor * silu);
