# Qwen3.8-27B on one V100, with speculative decoding

This is the **`exp-27b`** branch: llama.cpp with this project's changes, serving Qwen3.8-27B (Unsloth's `UD-Q4_K_XL` quantization) on a single V100 32 GB. The repository root's `README.md` is upstream llama.cpp's own and does not describe this fork.

## What this is

It decodes with speculative decoding, where a small draft model proposes several tokens and the 27B model checks them in one pass. There are two configurations:

- **MTP**: Unsloth's multi-token-prediction head as the draft.
- **DFlash2**: z-lab's DFlash2 draft model.

Both use rejection sampling (drafts are accepted or rejected in a way that keeps the model's sampling distribution) and a draft vocabulary (the draft only scores the 98,304 most likely token ids, which makes it cheaper). On this branch rejection sampling and the 98,304-id vocabulary are **engine defaults** rather than launch settings; `LLAMA_SPEC_REJECTION=0` and `LLAMA_SPEC_DRAFT_VOCAB=-1` restore the upstream behaviour.

The branch is the result of four rounds of kernel work on top of upstream llama.cpp. What they changed, and what each was worth, is under **Measured speed** below.

## Hardware and software it was built and measured on

- Tesla V100 32 GB PCIe, NVIDIA driver 580, CUDA 12.8, Ubuntu 26.04, gcc/g++ 14.
- The CUDA kernels were written and tested for that card only (`sm_70`). Other GPUs are untested and are not expected to work.

## Build

From the repository root, on the `exp-27b` branch. `ninja`, `cmake`, `gcc-14` and CUDA 12.8 must be installed.

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON
cmake --build build --target llama-server -j 12
```

The binary is `build/bin/llama-server`. On 12 CPU threads the build took about 9 minutes here, with ccache on.

## Models

Three files, about 20 GB in all. They go flat in `./models`, next to the draft-vocabulary file that is already in the repository.

| File | Repository | Size |
|---|---|---|
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | `unsloth/Qwen3.8-27B-GGUF` | 17,559,178,144 bytes (16.4 GiB) |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | `unsloth/Qwen3.8-27B-GGUF`, in `MTP/` | 1,369,590,656 bytes (1.3 GiB) |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | `z-lab/Qwen3.8-27B-DFlash2-GGUF` | 1,143,006,816 bytes (1.1 GiB) |

```
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf --local-dir models
mv models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf models/ && rmdir models/MTP
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models
```

`models/draft-vocab-qwen3.8-27b.txt` is the draft vocabulary: 131,072 token ids ranked by how often the 27B model and its sibling produce them on prose and code; this branch's default is to use the first 98,304. It ships in the repository.

## Run

```
scripts/serve-27b.sh mtp
scripts/serve-27b.sh dflash
```

Run them from the repository root. An optional second argument names the model directory (default `./models`). `LLAMA_HOST` (default `127.0.0.1`), `LLAMA_PORT` (default `8080`) and `LLAMA_BIN` set the address and the binary. On a machine with several cards, pick one with `CUDA_VISIBLE_DEVICES`. The server holds about 26 to 29 GB of the card's memory.

The two command lines the script runs, with `$MODELS` for the model directory. The script sets `LLAMA_SPEC_DRAFT_VOCAB_FILE`; rejection sampling and the vocabulary size are engine defaults, so they appear here only to show what is in force.

```
LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server --host 127.0.0.1 --port 8080 -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 7 -c 131072 -fa on -ngl 99 -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 2048 --jinja --metrics --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20
```

```
LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt LLAMA_DFLASH2_HEAD_FILE=$MODELS/mtp-Qwen3.8-27B-Q4_0.gguf build/bin/llama-server --host 127.0.0.1 --port 8080 -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 -c 131072 -fa on -ngl 99 -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 2048 --jinja --metrics --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20
```

The DFlash2 configuration also needs the MTP file: its output head reads that file's draft-vocabulary rows.

## Recommended settings

These are the settings that produced the figures below, and the ones we suggest for general purpose work — coding agents, Python and prose alike:

- **Sampling**: temperature 1.0, top-p 0.95, top-k 20.
- **Thinking**: on, at the chat template's default (`--jinja`), with `supportsReasoningEffort` left false in the client so the template's default applies.
- **Context**: 131,072 tokens, KV cache in f16 for both the target and the draft. The server fits in 26 to 29 GB of the card.
- **Prompt processing**: micro-batches of 2,048 tokens (`-b 4096 -ub 2048`), which is what the prefill figures below were measured with.
- **Concurrency**: one request at a time (`--parallel 1`).
- **Drafting**: a 7-token draft window (`--spec-draft-n-max 7`), with rejection sampling and the 98,304-id draft vocabulary on by default.
- **Model placement**: all layers on the GPU (`-ngl 99`), Flash Attention on (`-fa on`), 12 host threads (`-t 12`).

Nothing else needs tuning. `scripts/serve-27b.sh mtp` or `scripts/serve-27b.sh dflash` applies all of the above.

## Using it from a coding agent

This is the OpenAI-compatible provider block our coding agent uses. Replace `@PORT@` with the server's port. `supportsReasoningEffort` must stay false, so that the template's thinking default applies.

```json
{
  "providers": {
    "local": {
      "baseUrl": "http://127.0.0.1:@PORT@/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8-27B (single V100)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 32768,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "compat": {
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": false,
            "supportsStore": false,
            "supportsUsageInStreaming": true,
            "maxTokensField": "max_tokens"
          }
        }
      ]
    }
  }
}
```

## Measured speed

All on one V100 at about 1,380 MHz, single stream, with the recommended settings above. Figures that compare engines come from the same tasks, the same sampling and the same aggregation rule — total output tokens over total decode time.

**This branch, MTP, against its own previous release:**

| | previous release | this branch |
|---|---|---|
| Coding-agent suite, 4 tasks × empty and preloaded, every run at full marks | 123.7 / 110.3 tok/s | **126.6 / 111.9 tok/s** |
| Suite wall time | 14.2 min | **13.5 min** |
| HumanEval 0–31, mean steady decode | 103.40 tok/s, 31 of 32 with one token-cap loss | **107.70 tok/s, 32 of 32, none capped** |
| Cold prefill, fixed prompt, 16K / 64K | 1.4% to 2.6% lower | **+1.4% to +2.6% / +2.1% to +2.6%** |

**Against NInfer on the same four tasks, once each, empty and preloaded**, both engines solving every run:

| | decode tok/s | acceptance | tokens per round | prefill tok/s |
|---|---|---|---|---|
| this branch (MTP, draft window 7) | **115.6** | 0.599–0.668 | 5.19–5.68 | 482–670 |
| NInfer (its published settings, draft window 3) | **57.8** | 0.646–0.714 | 3.21–3.57 | **557–734** |

NInfer's acceptance is higher and its **prefill is faster**; its tokens per round are lower because it drafts 3 tokens where this branch drafts 7, so those two columns are not like-for-like. NInfer's published 262,144-token context does not fit a 32 GB V100 — its startup reservation needs 11.7 GB beyond the weights against 1 GB of automatic headroom and 12.6 GB free — so that run used 131,072. This branch serves 131,072 on the same card in 26 to 29 GB.

**What each round of kernel work was worth** (rounds at 64K context unless stated): the verification step reading each cached KV head once, −13%; the 8-row attention kernel widened to 2 to 8 rows, −21% at a 5-token draft window; the streaming attention kernel at four columns per warp, −2.6 ms per round at 64K and −5.1 at 120K; the device-side attention mask, −1.5 ms; the four-column gated-delta decode recurrence, −0.23 to −0.30 ms; the fused decode kernel, −0.56 to −0.62 ms; the chunked gated-delta prefill, +9.0% at 16K and +7.5% at 64K; and the checkpoint fix, −241 to −511 ms per request. Several other changes were measured and did not pay for themselves; they are not in this branch.

Speed depends on the task, the context length and the temperature, and these figures are one seed each.

## Known limits

- **Prefill is the slow side**, and NInfer prefills faster than this branch (557 to 734 tok/s against 482 to 670 on the same tasks).
- The draft vocabulary is tuned for English and code. Acceptance on Chinese is low.
- Context shift is off, so a full context cuts a reply short.
- The coding-agent figures are one seed per run; the model is sampled, so turn counts and paths differ between runs, and per-task figures are not like-for-like across engines.
- Untested: the `hf download` lines above were written from the local copies' repository ids and were not re-run from an empty directory; any card other than the V100, any driver or CUDA version other than the ones above, and more than one request at a time.
