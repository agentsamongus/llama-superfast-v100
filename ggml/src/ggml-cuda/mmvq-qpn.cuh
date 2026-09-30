#pragma once

#include "common.cuh"

// Volta tensor-core products for the dense K-quants, the 4-bit codebook types and Q8_0, on weights repacked into fragment
// order at load. LLAMA_MMVQ_QPN: atoi bitmask of types, default all on; 0 repacks nothing, so
// every product keeps the GGUF layout and the dp4a/MMQ/cuBLAS paths (7 = the K-quants only, as before; 31 = the
// types before).
// Only the shapes in the routing table are repacked, unless LLAMA_MMVQ_QPN_FORCE=1 (every eligible weight of an enabled
// type, for study; any other nonzero value forces only the types of that bitmask).
#define GGML_CUDA_QPN_Q5_K   1
#define GGML_CUDA_QPN_Q4_K   2
#define GGML_CUDA_QPN_Q6_K   4
#define GGML_CUDA_QPN_IQ4_XS 8
#define GGML_CUDA_QPN_IQ4_NL 16
#define GGML_CUDA_QPN_Q3_K   32
#define GGML_CUDA_QPN_Q8_0   64
#define GGML_CUDA_QPN_ALL    (GGML_CUDA_QPN_Q5_K | GGML_CUDA_QPN_Q4_K | GGML_CUDA_QPN_Q6_K | GGML_CUDA_QPN_IQ4_XS | GGML_CUDA_QPN_IQ4_NL | \
                              GGML_CUDA_QPN_Q3_K | GGML_CUDA_QPN_Q8_0)

// products of at most this many tokens run the tensor-core kernel (in passes of 8); larger ones dequantize to fp16 for cuBLAS
#define GGML_CUDA_QPN_MAX_TOKENS 32

// true if t (or the tensor it views) holds the repacked layout; only ggml_cuda_mul_mat_qpn and the prefill path may read it
static inline bool ggml_cuda_qpn_is_repacked(const ggml_tensor * t) {
    return t != nullptr && ((t->flags | (t->view_src ? t->view_src->flags : 0)) & GGML_TENSOR_FLAG_BACKEND_LAYOUT) != 0;
}

// load time (via the backend registry's proc address "ggml_backend_cuda_qpn_repack"): repacks t in place and tags it
// when its type is enabled and its shape is routed; returns true if it did
bool ggml_backend_cuda_qpn_repack(ggml_tensor * t);

// t's type is enabled, t is a plain 2-D weight the kernels can take, and its shape is routed (or LLAMA_MMVQ_QPN_FORCE)
bool ggml_cuda_qpn_eligible(const ggml_tensor * t, int cc);
// the same for a draft model's weight (its MTP layer, its LM head subset), by the draft's own route table
bool ggml_cuda_qpn_eligible_draft(const ggml_tensor * t, int cc);
// ... and the load-time repack of such a weight, via the backend registry's proc address "ggml_backend_cuda_qpn_repack_draft"
bool ggml_backend_cuda_qpn_repack_draft(ggml_tensor * t);
// the same for a DFlash2 draft's weights, by its own route table (proc address "ggml_backend_cuda_qpn_repack_dflash")
bool ggml_cuda_qpn_eligible_dflash(const ggml_tensor * t, int cc);
bool ggml_backend_cuda_qpn_repack_dflash(ggml_tensor * t);
// repacks an eligible t in place on device and tags it (synchronous; uses a temporary of at most 64 MiB)
bool ggml_cuda_qpn_repack(ggml_tensor * t, int device);

// the GGUF bytes of a repacked t, all of them, into host memory of ggml_nbytes(t) (synchronous; uses a temporary of at
// most 64 MiB); false if t is not repacked (an LM head subset gathered from a repacked head)
bool ggml_cuda_qpn_unpack(const ggml_tensor * t, void * host_dst, int device);
// ... the same, via the backend registry's proc address "ggml_backend_cuda_qpn_unpack"
bool ggml_backend_cuda_qpn_unpack(const ggml_tensor * t, void * host_dst);

// y = W x for a repacked W and 1..GGML_CUDA_QPN_MAX_TOKENS tokens
void ggml_cuda_mul_mat_qpn(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// the whole repacked W as fp16, row-major, the same values ggml's to_fp16 gives for the GGUF layout (prefill path)
void ggml_cuda_qpn_to_fp16(const ggml_tensor * src0, half * dst, cudaStream_t stream);

// bytes of one pass's prepared input (fp16 fragments, per-32 sums, range scales) for K columns and T <= 8
// tokens; products of one pass that read the same input share it (ggml_cuda_plan_q8_share, LLAMA_QPN_SHARE)
size_t ggml_cuda_qpn_prep_bytes(int64_t K, int T);
// where xh, xs and xsc sit in a prepared-input buffer of ggml_cuda_qpn_prep_bytes(K, T) at base
void ggml_cuda_qpn_prep_ptrs(char * base, int64_t K, int T, half ** xh, half ** xs, float ** xsc);
// qpn_prep_kernel's launch, x (T columns of K) into such a buffer, the sums included (LLAMA_QPN_PREP_CHECK)
void ggml_cuda_qpn_prep(const float * x, int64_t stride_col_x, int64_t K, int T, char * base, cudaStream_t stream);
// defined in ggml-cuda.cu: the shared buffer of this product's input, or nullptr; *ready if a sibling already prepared it
char * ggml_cuda_qpn_share_buffer(const ggml_tensor * src0, const ggml_tensor * src1, size_t nbytes, cudaStream_t stream, bool * ready);

// two products on repacked weights that read the same src1 (3 or 4 tokens), as one launch; ggml-cuda.cu plans the
// pairs (LLAMA_QPN_GROUP, default on). _ok: the pair can be launched together (types, K, a configured shape pair)
bool ggml_cuda_qpn_group_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * src1);
void ggml_cuda_mul_mat_qpn2(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[2], const ggml_tensor * src1, ggml_tensor * const dsts[2]);

// three products on repacked weights that read the same src1 (the attention input: q + gate, k, v; 4 to 8 tokens), as one
// launch; ggml-cuda.cu plans them (LLAMA_QPN_GROUP, and LLAMA_QPN_WIDE bit 8). _ok: types instantiated and a configured shape triple
bool ggml_cuda_qpn_group3_ok(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * src1);
void ggml_cuda_mul_mat_qpn3(ggml_backend_cuda_context & ctx, const ggml_tensor * const src0s[3], const ggml_tensor * src1, ggml_tensor * const dsts[3]);
