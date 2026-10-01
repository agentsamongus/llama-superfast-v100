# Qwen3.8-27B on one V100, with speculative decoding

This fork of llama.cpp serves Qwen3.8-27B (Unsloth's `UD-Q4_K_XL` quantization) on a single Tesla V100 32 GB. The 27B work was done on the `exp-27b` branch and is merged into `main`. The repository root's `README.md` has the short version and, below its rule, upstream llama.cpp's own README.

## What this is

The server decodes with speculative decoding: a small draft model proposes several tokens and the 27B model checks them in one pass. There are two draft configurations:

- **MTP**: Unsloth's multi-token-prediction head as the draft. This is the one we use and the one every figure below was measured with.
- **DFlash2**: z-lab's DFlash2 draft model, supported as an alternative.

Both use rejection sampling, which accepts or rejects draft tokens in a way that keeps the model's own sampling distribution, and a draft vocabulary, which has the draft score only the 98,304 most likely token ids. Both are engine defaults on this fork rather than launch settings; `LLAMA_SPEC_REJECTION=0` and `LLAMA_SPEC_DRAFT_VOCAB=-1` restore the upstream behaviour.

## Hardware and software it was built and measured on

- Tesla V100 32 GB PCIe, NVIDIA driver 580, CUDA 12.8, Ubuntu 26.04, gcc/g++ 14.
- The CUDA kernels were written and tested for that card only (`sm_70`). Other GPUs are untested and are not expected to work.

## Build

From the repository root. `ninja`, `cmake`, `gcc-14` and CUDA 12.8 must be installed.

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON
cmake --build build --target llama-server -j 12
```

The binary is `build/bin/llama-server`. On 12 CPU threads the build took about 9 minutes here, with ccache on.

## Models

Three files, about 20 GB in all. They go flat in `./models`, next to the draft-vocabulary file that is already in the repository. Only the first two are needed for MTP.

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

`models/draft-vocab-qwen3.8-27b.txt` is the draft vocabulary: 131,072 token ids ranked by how often the 27B model and its sibling produce them on prose and code. The default is to use the first 98,304. It ships in the repository.

## Run

```
scripts/serve-27b.sh mtp
scripts/serve-27b.sh dflash
```

Run them from the repository root. An optional second argument names the model directory (default `./models`). `LLAMA_HOST` (default `127.0.0.1`), `LLAMA_PORT` (default `8080`) and `LLAMA_BIN` set the address and the binary. On a machine with several cards, pick one with `CUDA_VISIBLE_DEVICES`. The server holds about 26 to 29 GB of the card's memory.

These are the two command lines the script runs, with `$MODELS` for the model directory. The script sets `LLAMA_SPEC_DRAFT_VOCAB_FILE`; rejection sampling and the vocabulary size are engine defaults.

```
LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server --host 127.0.0.1 --port 8080 -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 7 -c 131072 -fa on -ngl 99 -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 2048 --jinja --metrics --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20
```

```
LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt LLAMA_DFLASH2_HEAD_FILE=$MODELS/mtp-Qwen3.8-27B-Q4_0.gguf build/bin/llama-server --host 127.0.0.1 --port 8080 -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 -c 131072 -fa on -ngl 99 -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 2048 --jinja --metrics --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20
```

The DFlash2 configuration also needs the MTP file: its output head reads that file's draft-vocabulary rows.

## Settings

These are the settings behind every figure below, and the ones we use for everyday work: coding agents, Python and prose alike.

- **Sampling**: temperature 1.0, top-p 0.95, top-k 20.
- **Thinking**: on, at the chat template's default (`--jinja`). The client must not send a reasoning-effort field, so that the template's default applies.
- **Context**: 131,072 tokens, KV cache in f16 for both the target and the draft. The server fits in 26 to 29 GB of the card.
- **Prompt processing**: micro-batches of 2,048 tokens (`-b 4096 -ub 2048`).
- **Concurrency**: one request at a time (`--parallel 1`).
- **Drafting**: a 7-token draft window (`--spec-draft-n-max 7`), with rejection sampling and the 98,304-id draft vocabulary on by default.
- **Placement**: all layers on the GPU (`-ngl 99`), flash attention on (`-fa on`), 12 host threads (`-t 12`).

Nothing else needs tuning. `scripts/serve-27b.sh mtp` applies all of the above.

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

Everything here is from one V100 32 GB PCIe, single stream, with the settings above, on the current build. The card ran at 1,345 to 1,380 MHz while decoding and at about 1,290 MHz during the long prompt-processing runs, where the power cap bites. Peak memory use was 29,832 MiB as reported by nvidia-smi. Each run is one sample at temperature 1.0.

### Coding-agent tasks

Four feature tasks on a small Python web application, each with tests that must pass, run by a coding agent that reads files, edits them and runs the tests. Each task runs twice: from an empty context, and with the repository loaded into context first, which puts about 72K tokens in the context before the task starts. "Decode" is the server's steady decode rate over the run; "acceptance" is accepted draft tokens over drafted tokens; "context" is the task's own context at the end of the run, on top of the preload where there is one. All eight runs passed their tests.

| Task | Mode | Decode tok/s | Acceptance | Context at end | Turns | Wall |
|---|---|---|---|---|---|---|
| 1 | empty | 123.1 | 0.622 | 30K | 10 | 2.0 min |
| 1 | preloaded | 108.4 | 0.617 | 72K + 18K | 49 | 4.5 min |
| 2 | empty | 127.2 | 0.639 | 23K | 12 | 1.3 min |
| 2 | preloaded | 117.3 | 0.667 | 72K + 9K | 36 | 3.2 min |
| 3 | empty | 124.4 | 0.622 | 24K | 16 | 2.0 min |
| 3 | preloaded | 113.4 | 0.654 | 72K + 18K | 69 | 4.6 min |
| 4 | empty | 131.6 | 0.680 | 32K | 17 | 1.8 min |
| 4 | preloaded | 108.5 | 0.607 | 72K + 14K | 45 | 4.2 min |

Mean decode: 126.6 tok/s from an empty context, 111.9 tok/s preloaded. Loading the 72K-token repository into context took 103 to 107 seconds of prompt processing per run; during that phase, on the agent's own incremental prompts, the server processed 600 to 645 tok/s.

### HumanEval

Problems 0 to 31, one sample each, scored on the base tests and on the extended HumanEval+ tests.

- 32 of 32 passed on both sets. No answer hit the length cap.
- 107.7 tok/s mean steady decode. Counting prompt processing and everything else, 92.1 tok/s over the whole run.
- 3.71 tokens accepted per round, pooled over all rounds. Mean answer length 1,615 tokens.

### Prompt processing

A fixed prompt, processed cold (no cached prefix), two launches, with the card at about 1,290 MHz. The spread between the two launches is 1.2% at 16K and 0.5% at 64K.

| Prompt | Launch 1 | Launch 2 |
|---|---|---|
| 16K tokens | 935 tok/s (17.5 s) | 924 tok/s (17.7 s) |
| 64K tokens | 742 tok/s (86.3 s) | 738 tok/s (86.7 s) |

### Day-to-day use

On the author's own coding traffic over about 21,600 rounds, the server reported 0.457 accepted per drafted token (33,584 of 73,446) and 2.56 tokens per round. That is well below the coding-agent tasks above (0.61 to 0.68) and HumanEval (3.71 per round). We do not know the cause yet. The adaptive draft width, which narrows the 7-token window when a full-width round does not pay, uses a cost table measured on an older version of the verification kernel and is the first suspect.

### Against NInfer

Measured on the previous build of this fork, which the current one beats by about 2% on the same tasks. The same four tasks, once each in both modes, both engines passing every run, both aggregated the same way: total output tokens over total decode time.

| | Decode tok/s | Acceptance | Tokens per round | Prompt processing tok/s |
|---|---|---|---|---|
| This fork (MTP, 7-token window) | 115.6 | 0.599 to 0.668 | 5.19 to 5.68 | 482 to 670 |
| NInfer (its published settings, 3-token window) | 57.8 | 0.646 to 0.714 | 3.21 to 3.57 | 557 to 734 |

NInfer's acceptance is higher and its prompt processing is faster. Its tokens per round are lower because it drafts 3 tokens where this fork drafts 7, so those two columns are not like for like. NInfer's published 262,144-token context does not fit a 32 GB V100: its startup reservation needs 11.7 GB beyond the weights, against 12.6 GB free, so that run used 131,072. This fork serves 131,072 on the same card in 26 to 29 GB.

## What was changed, and what each change was worth

Rounds are at 64K context unless stated. Every change was gated against the previous build: bitwise on the output text, or by KL divergence where a numeric path changed. Several other changes were measured, did not pay for themselves, and are not in the fork.

- The verification step reads each cached KV head once instead of twice: 13% off the round.
- The 8-row attention kernel widened to serve any width from 2 to 8 rows: 21% off the round at a 5-token draft window.
- A streaming attention kernel at four columns per warp, with its key loads moved off the QK warps: 2.6 ms off the round at 64K, 5.1 ms at 120K.
- The attention mask built on the device rather than copied every step: 1.5 ms off the round.
- A chunked gated-delta prefill on tensor cores: prompt processing up 9.0% at 16K and 7.5% at 64K.
- The gated-delta decode recurrence at four columns per warp: 0.23 to 0.30 ms off the round.
- A fused decode kernel: 0.56 to 0.62 ms off the round.
- A checkpoint fix that stopped the draft model's KV cache being copied on every request: 241 to 511 ms off each request.
- The deployment's environment settings moved out of the launch line and into the engine as defaults.

## Known limits

- Prompt processing is the slow side, and NInfer prompts faster than this fork (557 to 734 tok/s against 482 to 670 on the same tasks).
- Draft acceptance in day-to-day use is lower than on the benchmarks (see above).
- The draft vocabulary is tuned for English and code. Acceptance on Chinese is low.
- Context shift is off, so a full context cuts a reply short.
- Every figure is one sample per run at temperature 1.0; turn counts and paths differ between runs, and per-task figures are not like for like across engines.
- Untested: the `hf download` lines above were written from the local copies' repository ids and were not re-run from an empty directory; any card other than the V100, any driver or CUDA version other than the ones above; and more than one request at a time.
