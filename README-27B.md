# llama-superfast-v100: Qwen3.8-27B on one V100, with speculative decoding

## What this is

This is llama.cpp with this project's changes, serving Qwen3.8-27B (Unsloth's `UD-Q4_K_XL` quantization) on a single V100 32 GB. It decodes with speculative decoding, where a small draft model proposes several tokens and the 27B model checks them in one pass. There are two configurations:

- **MTP**: Unsloth's multi-token-prediction head as the draft.
- **DFlash2**: z-lab's DFlash2 draft model.

Both use rejection sampling (drafts are accepted or rejected in a way that keeps the model's sampling distribution) and a draft vocabulary (the draft only scores the 98,304 most likely token ids, which makes it cheaper). The rest of `README.md` is upstream llama.cpp's and is unchanged; this file covers only the 27B build.

## What this repository is

This is a fork of [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp), based on upstream commit `6ba30d0`. It was **not cloned from upstream**: it was imported from the source tree that Unsloth's `b11030-mix-5ff778e` release asset was built from (archive sha256 `44cca07c…`), so the root commit here is that import and there is no upstream history behind it. For a reader comparing the two: diff this tree against upstream at `6ba30d0` and the difference is our work, plus whatever Unsloth's build of that commit carried that upstream's did not. We did not separate the two, so treat the import commit as the baseline and the commits after it as ours.

## Hardware and software it was built and measured on

- Tesla V100 32 GB PCIe, NVIDIA driver 580, CUDA 12.8, Ubuntu 26.04, gcc/g++ 14.
- The CUDA kernels were written and tested for that card only (`sm_70`). Other GPUs are untested and are not expected to work.

## Build

From the repository root, on the `public-27b` branch (or the `v1.0-27b` tag). `ninja`, `cmake`, `gcc-14` and CUDA 12.8 must be installed.

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON
cmake --build build --target llama-server -j 12
```

The binary is `build/bin/llama-server`. The build also downloads the server's web UI bundle from Hugging Face (`ggml-org/llama-ui`), so it needs network access. On 12 CPU threads the build took about 9 minutes here, with ccache on.

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

`models/draft-vocab-qwen3.8-27b.txt` is the draft vocabulary: 131,072 token ids ranked by how often they occur in the 27B model's own outputs, in a sibling model's outputs (Qwen3.8-Flash-Next, same tokenizer), in general prose and in code; the server uses the first 98,304. Its provenance is in "Licences and credits" below. It ships in the repository.

## Run

```
scripts/serve-27b.sh mtp
scripts/serve-27b.sh dflash
```

Run them from the repository root. An optional second argument names the model directory (default `./models`). `LLAMA_HOST` (default `127.0.0.1`), `LLAMA_PORT` (default `8080`) and `LLAMA_BIN` set the address and the binary. On a machine with several cards, pick one with `CUDA_VISIBLE_DEVICES`. The server holds about 28 to 29 GB of the card's memory.

The two command lines the script runs, with `$MODELS` for the model directory:

```
LLAMA_SPEC_REJECTION=1 LLAMA_SPEC_DRAFT_VOCAB=98304 LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server --host 127.0.0.1 --port 8080 -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 7 -c 131072 -fa on -ngl 99 -ctk bf16 -ctv bf16 -ctkd bf16 -ctvd bf16 -t 12 -b 2048 -ub 512 --jinja --metrics --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20
```

```
LLAMA_SPEC_REJECTION=1 LLAMA_SPEC_DRAFT_VOCAB=98304 LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt LLAMA_DFLASH2_HEAD_FILE=$MODELS/mtp-Qwen3.8-27B-Q4_0.gguf build/bin/llama-server --host 127.0.0.1 --port 8080 -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 -c 131072 -fa on -ngl 99 -ctk bf16 -ctv bf16 -ctkd bf16 -ctvd bf16 -t 12 -b 2048 -ub 512 --jinja --metrics --parallel 1 --temp 1.0 --top-p 0.95 --top-k 20
```

The DFlash2 configuration also needs the MTP file: its output head reads that file's draft-vocabulary rows.

## Settings we use

- Sampling: temperature 1.0, top-p 0.95, top-k 20.
- Thinking on, at the chat template's default (`--jinja`).
- Context 131,072 tokens, with the KV cache in bf16, for the target and the draft.
- `--parallel 1`: one request at a time.

## Using it from a coding agent

This is the Pi `models.json` provider block we use. Replace `@PORT@` with the server's port. `supportsReasoningEffort` must stay false, so that the template's thinking default applies.

```json
{
  "providers": {
    "local-27b": {
      "baseUrl": "http://127.0.0.1:@PORT@/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8-27B (DFlash2, card 0)",
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

All on one V100 at 1,380 MHz, with the settings above.

- HumanEval problems 0 to 31, single stream, replies capped at 16K tokens: DFlash2 124.4 tok/s mean steady decode, MTP 101.1 (scripts and conditions in "Reproducing the numbers" below).
- Coding-agent runs of 2026-09-30 with DFlash2, four small tasks: 82.9 to 112.2 tok/s decode over whole runs that peaked at 20K to 35K tokens of context; 69.8 to 84.8 tok/s when the run first read 65 files (peak context 82K to 89K), a figure that includes those reading turns. Prefill (reading the prompt) ran at 415 to 459 tok/s in the runs where it was recorded.
- Draft acceptance on those runs was 0.42 to 0.55 (short context) and 0.51 to 0.59 (long).

Speed depends on the task, the context length and the temperature; these figures are from those runs only.

## Known limits

- The draft vocabulary is tuned for English and code. Measured against NInfer's held-out chat traffic, the shipped list at 98,304 ids covers 98.95% of English chat's accepted draft positions, 93.6% of code's, 74.3% of other-language chat's and 69.2% of Chinese chat's, so expect weaker drafting on Chinese and other non-English text. (Coverage is the share of positions whose token is inside the subset; it is not a measured speed.)
- Context shift is off, so a full context cuts a reply short.
- Prefill is the slow side.
- Untested: the `hf download` lines above were written from the local copies' repository ids and were not re-run from an empty directory; any card other than the V100, any driver or CUDA version other than the ones above, and more than one request at a time.

## Reproducing the numbers

Every figure above was measured on one V100 at 1,380 MHz (the SM clock stayed there in 344 of 348 samples in the DFlash2 run and 533 of 536 in the MTP run), one request at a time, with the launch lines under "Run".

### HumanEval+ (the two tok/s figures above)

`scripts/humaneval/eval.py` sends HumanEval problems 0 to 31 to a running server, one at a time, and scores them with EvalPlus. It uses EvalPlus's own chat instruction and its own answer sanitizer, temperature 1.0, top-p 0.95, top-k 20, seed 36, thinking on (the chat template's default, no `reasoning_effort` sent) and a 16,384-token cap on each reply.

```
pip install evalplus==0.3.1
scripts/serve-27b.sh mtp > server.log 2>&1 &
python3 scripts/humaneval/eval.py 8080 server.log 36
```

Stop the server and repeat with `scripts/serve-27b.sh dflash > server.log 2>&1 &` for the other configuration. `server.log` must be the file the server writes to, because the script reads tokens per round from it. It writes `progress.txt`, `stage-8-summary.*` (problems 0 to 7) and `stage-32-summary.*` (problems 0 to 31) next to itself, and one JSON per problem under `run-seed36/`. On Python 3.14 the script sets the `fork` start method, because the default breaks EvalPlus's checker.

What we measured, on problems 0 to 31, one seed (so 32 samples; the pass rates carry a wide margin):

| | DFlash2 | MTP |
|---|---|---|
| pass@1 base / plus | 32/32 / 31/32 | 32/32 / 31/32 |
| mean steady decode, tok/s (mean of the per-request rates) | 124.38 | 101.07 |
| aggregate output, tok/s (all output tokens / all decode time) | 112.93 | 88.67 |
| tokens per round, pooled | 4.269 | 3.731 |
| mean output length | 1,051 tokens | 1,428 tokens |

The plus miss in both runs is HumanEval/22. The two runs sample differently at the same seed, so they wrote different texts; tok/s is per token and compares, but the output lengths do not make a like-for-like speed contrast. Two conditions the figures depend on: the DFlash2 run had the per-round timers switched on (`LLAMA_ROUND_TIMERS=1`), which adds a little overhead, and the MTP run had a compile job running on the same machine for its last minute, so its last two problems are indicative only.

### The coding-agent smoke (the "Coding-agent runs" figures above)

This is a method we describe but do not ship as something you can run, for two reasons: the tasks are written against the author's own web application (a photo indexer), and the runner is tied to the author's machine layout (a sandbox, and one model server per card). If you want to repeat it on your own code base, this is what was done:

- **Agent:** the Pi coding agent (`--mode json`), pointed at the server through the provider block under "Using it from a coding agent", with `supportsReasoningEffort` false, in a sandbox that could write only its own copy of the repository.
- **Task:** a written spec ("add star ratings to the web backend", and three more of the same size) plus a test suite copied into the run's `tests/` directory. The agent was told to make `pytest tests -q -x --tb=short` pass, not to edit tests, and to stop with a one-line reply once it did. The score is the pass count of the original test files, run outside the agent's sandbox; a run also records whether the agent modified tests (none did).
- **Empty mode:** the agent starts with only the task. **Preload mode:** it first reads 65 files of the repository in full (about 71K to 72K tokens of prompt before the task), and the task's own context is counted from the end of that reading. A flag is raised at 40K tokens of the task's own context and the run is stopped at 100K; no run reached either.
- **Sampling and server:** as under "Settings we use"; DFlash2 unless stated.
- **What came out (single samples at T=1, so no A/A spread):** all four tasks scored full marks in both modes (13/13, 12/12, 13/13 and 15/15). Empty mode: 82.9 to 112.2 tok/s decode over the whole run, the task's context 20K to 35K, 1.5 to 3.3 minutes. Preload mode: 69.8 to 84.8 tok/s (reading turns included), the task's own context 8.7K to 17.1K, 1.6 to 3.3 minutes for the task after 3.0 to 3.4 for the reading. On the ratings task with MTP in place of DFlash2: 98.3 tok/s and acceptance 0.674 in empty mode against 82.9 and 0.419, and 65.0 tok/s in preload mode against 69.8. In preload mode MTP read the files in many small steps (78 requests against 18), so its wall time was longer (460 s against 351 s); one sample cannot say how much of that is the model's choices rather than the speculation mode.

## Tests

The six test programs we added live in `tests/` and are built with the rest of the tests. Configure with `-DLLAMA_BUILD_TESTS=ON` added to the build command above, then:

```
cmake --build build --target test-mmvq-tc test-qsa-compact test-qsa-host-meta test-qsa-select test-spec-draft-grammar test-spec-rejection -j 12
ctest --test-dir build -R 'test-(mmvq-tc|qsa-compact|qsa-host-meta|qsa-select|spec-draft-grammar|spec-rejection)' --output-on-failure
```

Each test also runs by itself as `build/bin/<name>`.

| Test | Needs a GPU | What it checks |
|---|---|---|
| `test-mmvq-tc` | yes, first CUDA device | The tensor-core path for few-token K-quant products: accuracy against an fp64 reference at 3 to 8 tokens; activations up to 1e6 stay finite and equally accurate; a token holding inf or NaN behaves as on the dp4a path (compared with a child process run with `LLAMA_MMVQ_TC=0`) without disturbing other tokens; and a tensor-core product sharing an activation quantization with dp4a products neither leads nor joins their group. |
| `test-qsa-compact` | yes | Differential test of the compact selected-KV attention for the `qwen4exp` architecture against dense flash attention with the same mask, on the same bf16 cache. Set `QSA_TEST_PER_TOKEN` for per-token output. |
| `test-qsa-host-meta` | no | The incremental block table used for block selection against a full scan, on random cache histories (appends, rollbacks, prefix reuse, copies, shifts, restores, clears), byte for byte. Run `build/bin/test-qsa-host-meta bench` to time both at 20K, 64K and 128K instead. |
| `test-qsa-select` | runs the CPU backend and every GPU backend | The block-selection op against a brute-force reference, bit for bit, including ties at the k-th place, all-zero scores and rows with fewer selectable blocks than k, and repeatability on the GPU. |
| `test-spec-draft-grammar` | no | A drafted token that does not fit the target's tool-call grammar must not throw when it is accepted into the sampler's copy (`common_sampler_accept_draft`); the control shows the plain accept does throw. It loads only a vocabulary file, `models/ggml-vocab-qwen35.gguf`, which `ctest` passes; run by hand as `build/bin/test-spec-draft-grammar models/ggml-vocab-qwen35.gguf`. |
| `test-spec-rejection` | no | The rejection step of speculative sampling keeps the target's distribution: 10^6 draws per synthetic pair of distributions, a chi-square p-value above 0.01, no token outside the target's support, and the acceptance rate printed beside its expected value. Also covers the adaptive draft length. |

We did not run these when preparing this repository for publication. What we did check: all six test programs and `llama-server` built without error from this tree (381 build steps, exit status 0).

## Licences and credits

- **This repository:** upstream llama.cpp's MIT licence (`LICENSE`, "Copyright (c) 2023-2026 The ggml authors") is kept as it was, and our changes are released under the same licence. `licenses/` holds upstream's third-party notices (the vendored nlohmann JSON library's, for one).
- **Qwen3.8-27B, Unsloth's GGUFs:** the quantized model `Qwen3.8-27B-UD-Q4_K_XL.gguf` and the MTP head `mtp-Qwen3.8-27B-Q4_0.gguf` are Unsloth's conversions (`unsloth/Qwen3.8-27B-GGUF`) of Qwen's `Qwen/Qwen3.8-27B`. The GGUF metadata of both files says `general.license = apache-2.0` and `general.quantized_by = Unsloth`. No separate licence file for them was on the machine this was prepared on, so the metadata is the only evidence we have; check the model card before you rely on it.
- **DFlash2 draft:** `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` is z-lab's GGUF conversion (`z-lab/Qwen3.8-27B-DFlash2-GGUF`, a mirror of `incoai/Qwen3.8-27B-DFlash2-GGUF`) of Inco AI's DFlash 2 draft model for Qwen3.8-27B. The repository's README front matter and the file's GGUF metadata both say `apache-2.0`. The project link in that README is https://github.com/z-lab/dflash.
- **Draft vocabulary (`models/draft-vocab-qwen3.8-27b.txt`):** a ranking of token ids, no text. It was built by counting tokens in the 27B model's own replies (weighted four times per token), in replies from Qwen3.8-Flash-Next (a sibling model with the same tokenizer), in prose (a wiki-text evaluation corpus we did not record the licence of, this repository's documentation and our own notes) and in code (this repository's C++ and CUDA and the Python standard library). Only the counts were used, and no text is shipped. At 98,304 ids it covered at least 99.5% of tokens on the held-out job types we tried (poetry, CSV and other prompts). It does not use NInfer's data, so it needs no NInfer credit. A second, merged list that did use NInfer's token counts (Apache-2.0, https://github.com/neroued/ninfer-v100) was tried and is not shipped.
- **NInfer:** the design of our tensor-core products for K-quants follows the QPN kernels in NInfer (`ninfer-v100`, Apache-2.0), which in turn credits the "v100-skinny" project. The header comment of `ggml/src/ggml-cuda/mmvq-qpn.cu` says so, and states that no NInfer code was copied.
