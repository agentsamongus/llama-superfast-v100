#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    // the full-vocabulary logits of output i, even when backend samplers returned candidates for it
    float * get_logits_full_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);

    // [TAG_DFLASH2_FEAT_DEV] (ticket 0101), see llama_cparams::layer_inp_dev_rows and inject_dev
    void    set_embeddings_layer_inp_dev(int32_t n_rows_max);
    int32_t get_embeddings_layer_inp_dev_rows() const { return layer_inp_dev_n; }
    void    set_inject_dev(bool value) { cparams.inject_dev = value; }
    // copies the device-held input of the slot-th enabled layer (ascending layer order) into dst at byte offset dst_offs, on this
    // context's stream, and makes backend_dst's stream wait for it (ggml_backend_tensor_copy_async); false if not held
    bool    copy_layer_inp_dev(uint32_t slot, const ggml_tensor * dst, size_t dst_offs, ggml_backend_t backend_dst) const;
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // [TAG_LOGITS_DEFER] copy the raw logits still on the device into the host rows (all of them when row < 0)
    void logits_dev_fetch(int64_t row);

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    // [TAG_SPEC_PIPELINE] see llama-ext.h
    void pipe_select(int32_t slot);
    void pipe_decode_flags(bool async, bool backup);
    void pipe_drain();
    void pipe_hold_next(bool hold);
    bool pipe_resume();
    void pipe_abort();
    bool pipe_rows_independent(int64_t n_more) const;

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    // [TAG_DFLASH2_FEAT_DEV] (ticket 0101) the last decode's enabled layer-input tensors (ascending layer order) and their backend, when
    // it was one ubatch of layer_inp_dev_n <= cparams.layer_inp_dev_rows tokens; layer_inp_dev_n = 0 when they went to the host instead
    std::vector<ggml_tensor *> layer_inp_dev_t;
    ggml_backend_t             layer_inp_dev_backend = nullptr;
    int32_t                    layer_inp_dev_n = 0;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // [TAG_LOGITS_DEFER] when every output of a ubatch has a backend sampler, its raw logits are not copied to the
    // host. the last such ubatch's logits stay in its graph output until the next decode, and are fetched per row on
    // request (get_logits_full_ith). earlier ubatches are copied before the next ubatch runs. LLAMA_BACKEND_TOPK=0
    // restores the upstream behaviour, where those raw logits are never copied.
    struct logits_dev_info {
        ggml_backend_t backend = nullptr;
        ggml_tensor *  t       = nullptr; // t_logits of the pending ubatch
        int64_t        row0    = 0;       // its first output row
        int64_t        n       = 0;       // its number of output rows
    };

    bool logits_defer = true;

    logits_dev_info logits_dev;

    // per output row: the row in logits_dev.t that holds its logits, or -1 when the host row is valid
    std::vector<int32_t> logits_dev_row;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    // [TAG_SPEC_PIPE_LOOSE] the QSA record of the latest decode's graph (llama_pipe_rows_independent)
    llm_graph_result::qsa_record qsa_last;

    // [TAG_SPEC_PIPELINE] the second output slot. the live members above (logits ... output_swaps, buf_output) hold
    // slot pipe_live, pipe_alt holds the other one; pipe_select swaps them
    struct pipe_output_slot {
        ggml_backend_buffer_ptr         buf_output;
        buffer_view<float>              logits     = {nullptr, 0};
        buffer_view<float>              embd       = {nullptr, 0};
        buffer_view<float>              embd_nextn = {nullptr, 0};
        std::vector<buffer_view<float>> embd_layer_inp;

        buffer_view<float>       s_logits     = {nullptr, 0};
        buffer_view<llama_token> s_sampled    = {nullptr, 0};
        buffer_view<float>       s_probs      = {nullptr, 0};
        buffer_view<llama_token> s_candidates = {nullptr, 0};
        std::vector<uint32_t>    s_logits_count;
        std::vector<uint32_t>    s_probs_count;
        std::vector<uint32_t>    s_candidates_count;

        logits_dev_info      logits_dev;
        std::vector<int32_t> logits_dev_row;

        std::map<llama_seq_id, std::vector<float>> embd_seq;

        uint32_t               n_outputs = 0;
        std::vector<int32_t>   output_ids;
        std::vector<swap_info> output_swaps;
    };

    pipe_output_slot pipe_alt;

    int32_t pipe_live   = 0;     // the slot held in the live members
    int32_t pipe_newest = 0;     // the slot of the latest decode
    bool    pipe_async  = false; // flags of the next decode
    bool    pipe_backup = false;
    bool    pipe_hold   = false;

    // a decode held before its card-1 half (ggml_backend_sched_set_hold): its outputs are extracted on resume
    struct pipe_held_info {
        bool               active = false;
        int32_t            slot   = 0;
        llm_graph_result * res    = nullptr;
        int64_t            n_out  = 0;
        int64_t            n_tok  = 0;
        bool               raw    = false;
    } pipe_held;

    // per slot: recorded after the slot's output copies, so reading an older slot waits only for its own decode
    ggml_backend_event_t pipe_ev[2]     = {nullptr, nullptr};
    ggml_backend_t       pipe_ev_be[2]  = {nullptr, nullptr};
    bool                 pipe_ev_set[2] = {false, false};

    // a device copy of the older slot's deferred raw logits ([TAG_LOGITS_DEFER]), which the next decode's graph
    // output overwrites
    ggml_context_ptr        pipe_side_ctx;
    ggml_backend_buffer_ptr pipe_side_buf;
    ggml_tensor *           pipe_side = nullptr;
    ggml_context_ptr        pipe_view_ctx[2];

    void pipe_swap();
    void pipe_save_alt_logits();
    void pipe_rs_backup(const llama_batch & batch);

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // ticket 0101: a DFlash2 draft's injection decodes (embd batches of target features) run on a scheduler and graph arena of their
    // own (swapped in for the decode), so the injection graph and the draft pass graph stay allocated side by side and each is reused
    // round after round instead of both being rebuilt and re-allocated every round. LLAMA_DFLASH2_INJ_ARENA=0 turns it off
    ggml_backend_sched_ptr sched_inj;
    llm_graph_result_ptr   gf_res_inj;
    llm_graph_result *     gf_res_inj_active = nullptr;
    bool                   inj_arena = false; // the injection's scheduler and arena are swapped in
    void inj_arena_swap();

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
