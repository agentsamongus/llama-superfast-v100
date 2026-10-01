#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hc_post_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * hc_post, ggml_tensor * rms_norm, ggml_tensor * mul_tensor);

// ticket 0056, LLAMA_FOLD_NORM_GATE: dst = sigmoid(z) * (rms_norm(x) * w), the GDN output's gated norm (silu(z) when
// the gate node is a SILU, ticket 0087)
void ggml_cuda_op_rms_norm_mul_sigmoid_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * mul_tensor, const ggml_tensor * sigmoid, ggml_tensor * dst);

// ticket 0087, LLAMA_FOLD_ADD_NORM: add = a + b, then mul = rms_norm(add) * w, both written
void ggml_cuda_op_add_rms_norm_mul(ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * rms_norm,
        ggml_tensor * mul_tensor);
