#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// Producer chains folded into a gated_delta_net launch (planned by ggml_cuda_plan_gdn_folds): the
// kernel reads a chain's raw input and repeats its arithmetic, and the chain's nodes are not launched.
// A null pointer leaves that input unfolded.
struct ggml_cuda_gdn_fold {
    // LLAMA_GDN_FUSE_QKNORM: q and k are the raw conv output views, L2-normalised as rms_norm -> scale
    const float * q = nullptr;
    const float * k = nullptr;
    int64_t sq1 = 0, sq2 = 0, sq3 = 0; // raw q/k strides in floats
    int     norm_ncols = 0;
    float   eps_q = 0.0f, eps_k = 0.0f;
    float   scale_q = 0.0f, bias_q = 0.0f, scale_k = 0.0f, bias_k = 0.0f;
    // LLAMA_GDN_FUSE_GATES: g = softplus(alpha + dt) * a and beta = sigmoid(beta), dt and a one value per head.
    // alpha and beta are read at their own strides in floats (head, token, sequence): laid out as g and beta are,
    // or views of one product that computes both (qwen35's merged a/b projection)
    const float * alpha = nullptr;
    const float * dt    = nullptr;
    const float * a     = nullptr;
    const float * beta  = nullptr;
    int64_t sa1 = 0, sa2 = 0, sa3 = 0; // alpha
    int64_t sr1 = 0, sr2 = 0, sr3 = 0; // beta
    // LLAMA_GDN_STATE_DIRECT: the state is read from its recurrent cache row, states + ids[seq] * state_row, not
    // from a gathered copy (one sequence only, so no other sequence writes the row while it is read)
    const float   * states    = nullptr;
    const int32_t * ids       = nullptr;
    int64_t         state_row = 0; // floats per cache row
};

// fold may be null (no chain folded)
void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                  const ggml_cuda_gdn_fold * fold = nullptr);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache,
                                              const ggml_cuda_gdn_fold * fold = nullptr);

// LLAMA_GDN_FUSE_GATES: g = softplus(alpha + dt) * a in one launch, when the decay gate is not folded
// into the gated_delta_net launch; alpha and g contiguous and alike, dt and a one value per row element
void ggml_cuda_op_gdn_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * alpha, const ggml_tensor * dt,
                           const ggml_tensor * a, ggml_tensor * g);
