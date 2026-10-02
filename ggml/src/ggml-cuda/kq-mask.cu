#include "kq-mask.cuh"

//. One thread per 8 cells: it applies the changed cells to the mirror, then writes the
// 8 mask values of every row (keep: 0x0000, drop: -inf 0xFC00, the host loop's llama_cast of 0.0f and -INFINITY).
// A cell is written to the mirror and read for the mask by the same thread, so the update needs no grid barrier.

struct kqm_args {
    int32_t n_tokens;
    int32_t n_kv;
    int32_t kv_size;
    int32_t n_cols;   // cells covered: max(n_kv, upd_lo + upd_n)
    int32_t use_2d;
    int32_t upd_lo;
    int32_t upd_n;    // <= KQM_MAX_UPD
    int32_t p [KQM_MAX_TOKENS];
    int32_t py[KQM_MAX_TOKENS];
    int32_t px[KQM_MAX_TOKENS];
    int32_t upd[3*KQM_MAX_UPD];
};

static_assert(sizeof(kqm_args) + 2*sizeof(void *) <= 4096, "kernel arguments over 4 KB");

template <bool vec>
static __global__ void kq_mask_kernel(const kqm_args a, uint16_t * __restrict__ mask, int32_t * __restrict__ cells) {
    const int j0 = (blockIdx.x*blockDim.x + threadIdx.x)*8;
    if (j0 >= a.n_cols) {
        return;
    }

    int32_t v[8];
    int32_t y[8];
    int32_t x[8];
    uint32_t have_yx = 0;

#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int j = j0 + e;
        const int u = j - a.upd_lo;
        v[e] = -1;
        y[e] = 0;
        x[e] = 0;
        if (j < a.kv_size && u >= 0 && u < a.upd_n) {
            v[e] = a.upd[u];
            y[e] = a.upd[KQM_MAX_UPD + u];
            x[e] = a.upd[2*KQM_MAX_UPD + u];
            have_yx |= 1u << e;
            cells[j]               = v[e];
            cells[a.kv_size + j]   = y[e];
            cells[2*a.kv_size + j] = x[e];
        } else if (j < a.n_kv) {
            v[e] = cells[j];
        }
    }

    if (j0 >= a.n_kv) {
        return;
    }

    for (int i = 0; i < a.n_tokens; ++i) {
        const int32_t p = a.p[i];

        uint16_t out[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            bool keep = v[e] >= 0 && v[e] <= p;
            if (keep && a.use_2d && v[e] == p) {
                if (!(have_yx & (1u << e))) {
                    y[e] = cells[a.kv_size + j0 + e];
                    x[e] = cells[2*a.kv_size + j0 + e];
                    have_yx |= 1u << e;
                }
                if (y[e] > a.py[i] || (y[e] == a.py[i] && x[e] > a.px[i])) {
                    keep = false;
                }
            }
            out[e] = keep ? 0x0000 : 0xFC00;
        }

        uint16_t * row = mask + (int64_t) i*a.n_kv;
        if (vec && j0 + 8 <= a.n_kv) {
            uint4 w;
            w.x = out[0] | ((uint32_t) out[1] << 16);
            w.y = out[2] | ((uint32_t) out[3] << 16);
            w.z = out[4] | ((uint32_t) out[5] << 16);
            w.w = out[6] | ((uint32_t) out[7] << 16);
            *(uint4 *) (row + j0) = w;
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                if (j0 + e < a.n_kv) {
                    row[j0 + e] = out[e];
                }
            }
        }
    }
}

void ggml_cuda_kq_mask(cudaStream_t stream, half * mask, int64_t n_kv, int32_t * cells, int64_t kv_size,
        const int32_t * p, const int32_t * py, const int32_t * px, int32_t n_tokens, bool use_2d,
        const int32_t * upd, int32_t upd_lo, int32_t upd_n) {
    GGML_ASSERT(n_tokens > 0 && n_tokens <= KQM_MAX_TOKENS);
    GGML_ASSERT(n_kv <= kv_size && upd_lo >= 0 && upd_n >= 0 && upd_lo + (int64_t) upd_n <= kv_size);

    kqm_args a;
    a.n_tokens = n_tokens;
    a.n_kv     = (int32_t) n_kv;
    a.kv_size  = (int32_t) kv_size;
    a.use_2d   = use_2d ? 1 : 0;
    a.upd_lo   = upd_lo;
    a.upd_n    = upd_n;
    for (int i = 0; i < n_tokens; ++i) {
        a.p [i] = p [i];
        a.py[i] = py[i];
        a.px[i] = px[i];
    }

    if (upd_n > KQM_MAX_UPD) {
        // many changed cells (the first mask, or after a prompt): one copy of each array, ordered before the kernel
        CUDA_CHECK(cudaMemcpyAsync(cells + upd_lo,             upd,           upd_n*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cells + kv_size + upd_lo,   upd + upd_n,   upd_n*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cells + 2*kv_size + upd_lo, upd + 2*upd_n, upd_n*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        a.upd_lo = 0;
        a.upd_n  = 0;
    } else {
        for (int u = 0; u < upd_n; ++u) {
            a.upd[u]                 = upd[u];
            a.upd[KQM_MAX_UPD + u]   = upd[upd_n + u];
            a.upd[2*KQM_MAX_UPD + u] = upd[2*upd_n + u];
        }
    }

    a.n_cols = (int32_t) std::max<int64_t>(n_kv, (int64_t) a.upd_lo + a.upd_n);

    const int threads = 128;
    const int blocks  = (int) ((a.n_cols + 8*threads - 1) / (8*threads));

    const bool vec = n_kv % 8 == 0 && ((uintptr_t) mask % 16) == 0;
    if (vec) {
        kq_mask_kernel<true> <<<blocks, threads, 0, stream>>>(a, (uint16_t *) mask, cells);
    } else {
        kq_mask_kernel<false><<<blocks, threads, 0, stream>>>(a, (uint16_t *) mask, cells);
    }
    CUDA_CHECK(cudaGetLastError());
}
