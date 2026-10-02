# llama-superfast-v100: llama.cpp for the Tesla V100

This is a fork of llama.cpp with CUDA kernel and server changes for the Tesla V100 (Volta, `sm_70`). It runs **Qwen3.8-27B on a single V100 32 GB** with speculative decoding, at the speeds below, and it has run a coding agent on real repositories for seven hours straight without a restart. The same engine also serves Qwen3.8-Flash-Next across two V100s; that path is in the tree but is not documented here.

Everything below the horizontal rule is upstream llama.cpp's own README. The build, the model files, the launch lines, the full measurements, how to reproduce them, the tests, the fork statement and the licences are in [`README-27B.md`](README-27B.md).

## What it runs

- Qwen3.8-27B in Unsloth's `UD-Q4_K_XL` quantization (16.4 GiB), with Unsloth's multi-token-prediction head (`Q4_0`, 1.3 GiB) as the draft model. z-lab's DFlash2 draft model is supported as an alternative.
- One V100 32 GB. A 131,072-token context with the KV cache in f16 fits in 26 to 29 GB.
- One request at a time, thinking on, temperature 1.0, top-p 0.95, top-k 20. These are the settings behind every figure here, and the ones we use day to day.

## How fast

All figures are from one V100 32 GB PCIe, single stream, at the settings above, with the card at 1,345 to 1,380 MHz while decoding and about 1,290 MHz during long prompt processing, where the power cap bites.

| | Result |
|---|---|
| Seven hours of agentic work on DeepSWE tasks (17 tasks from the public catalogue, real Go, TypeScript, Python and Rust repositories, a coding agent with context compaction near 98K tokens) | 99.4 tok/s mean decode over 3.0 million generated tokens, 7 of 17 resolved, 52 context compactions survived, no restart and no error on either of two servers |
| Coding-agent tasks, starting from an empty context (runs end at 22K to 32K tokens) | 126.6 tok/s mean decode, 8 of 8 runs passed their tests |
| The same tasks with the repository loaded into context first (about 72K tokens before the task starts) | 111.9 tok/s mean decode |
| HumanEval, problems 0 to 31 | 107.7 tok/s mean steady decode, 32 of 32 passed, 3.7 tokens accepted per round |
| Prompt processing, cold, fixed prompt | about 930 tok/s for a 16K prompt, about 740 tok/s for a 64K prompt |
| Draft acceptance | 0.61 to 0.68 per drafted token on the coding-agent tasks, 0.52 on the DeepSWE tasks, with a 7-token draft window |

The DeepSWE run is the longest thing we have done with it: two servers, one per card, 3,800 requests over seven hours, every task running up to the compaction trigger one to nine times and carrying on. The misses were the agent's, not the engine's: the unresolved tasks decoded at the same speed and several were a few verifier tests short.

The coding-agent tasks are four ordinary feature tasks on a small Python web application, run by a coding agent that edits files and runs the tests, each once from an empty context and once with the repository preloaded. Every run is one sample at temperature 1.0, so turn counts and paths differ from run to run; the figures are typical rather than best-case.

One figure that is not from a benchmark: on the author's own day-to-day coding traffic over about 21,600 rounds, acceptance was 0.46 per drafted token and 2.6 tokens per round, lower than on the tasks above. We have not measured why.

A comparison with NInfer on the same tasks is in `README-27B.md`. In short, this fork decodes about twice as fast on those tasks; on prompt processing the two are now close on agent traffic, and on a cold 16K prompt this fork is past NInfer's published figure for the same card (about 930 against 833 tok/s).

## What was changed

About thirty changes to the CUDA kernels and the server, made over several rounds of work on this one card: tensor-core products for the few-token matrix-vector step on quantized weights, fused decode kernels for the gated-delta layers, attention kernels that read the cache once per verification step and stream at depth, a chunked tensor-core prefill, and a speculative-decoding loop with rejection sampling, a draft vocabulary and pipelined drafting. Each was kept only if it was faster on the same test and produced the same text, or, where a numeric path changed, stayed within a KL-divergence bound against the previous build. Changes that did not pay for themselves were measured and left out. The list, with what each was worth, is in `README-27B.md`.

## Limits

- Context is 131,072 tokens on one card; a 262,144-token f16 cache does not fit beside the weights. On long agentic tasks that is the limit that binds (every DeepSWE task compacted at least once).
- The CUDA kernels were written and tested for the V100 only. Other GPUs are untested and are not expected to work.
- One request at a time. Concurrent requests are untested.
- The draft vocabulary is tuned for English and code; acceptance on Chinese is low.
- Context shift is off, so a full context cuts a reply short.

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
