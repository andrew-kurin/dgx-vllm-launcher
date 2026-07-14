# dgx-vllm-launcher

A validated Docker launcher for serving supported FP8 and NVFP4 models with vLLM.

## Supported variants

| Variant | Default model source | Optional preloaded checkpoint | Default MoE backend |
| --- | --- | --- | --- |
| `qwen36-fp8` | `Qwen/Qwen3.6-35B-A3B-FP8` | — | `triton` |
| `qwen36-nvfp4` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `Qwen3.6-35B-A3B-NVFP4` | `marlin` |
| `qwen36-27b-nvfp4` | `nvidia/Qwen3.6-27B-NVFP4` | `Qwen3.6-27B-NVFP4` | `(none)` |
| `qwen36-27b-nvfp4-dflash` **(experimental)** | `nvidia/Qwen3.6-27B-NVFP4` | — | `(none)` |
| `gemma4-nvfp4` | `nvidia/Gemma-4-26B-A4B-NVFP4` | `Gemma-4-26B-A4B-NVFP4` | `(none)` |
| `ornith-nvfp4` | `sakamakismile/Ornith-1.0-35B-NVFP4` | `Ornith-1.0-35B-NVFP4` | `(none)` |
| `mistral4-nvfp4` | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | `Mistral-Small-4-119B-2603-NVFP4` | `(none)` |
| `diffusion-gemma-nvfp4` | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | `diffusiongemma-26B-A4B-it-NVFP4` | `(none)` |
| `nemotron3-nano-omni-nvfp4` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `(none)` |

The DFlash variant is an experimental comparison profile built around a
third-party drafter that is still under development. Use the native-MTP variant
as the default Qwen 3.6 27B path.

The launcher provides:

- A single immutable launch plan resolved before Docker is changed
- Profile-driven model, reasoning, tool-calling, and performance settings
- Optional preloaded checkpoints with explicit Hugging Face fallback warnings
- Docker and image preflight checks
- Health polling against a real monotonic deadline
- Required warmup and smoke checks when enabled
- Safe secret forwarding without putting tokens in Docker command arguments
- Loopback API publishing by default, with an explicit bind-address override
- Per-launch container identity, collision protection, and foreground cleanup

## Quick start

```bash
cd path/to/dgx-vllm-launcher
uv sync --group dev

uv run dvl qwen36-fp8
uv run dvl qwen36-nvfp4
uv run dvl qwen36-27b-nvfp4
# Experimental comparison mode
uv run dvl qwen36-27b-nvfp4-dflash
uv run dvl gemma4-nvfp4
uv run dvl ornith-nvfp4
uv run dvl mistral4-nvfp4
uv run dvl diffusion-gemma-nvfp4
uv run dvl nemotron3-nano-omni-nvfp4
```

The full command and shorter entrypoints are equivalent:

```text
dgx-vllm-launcher
dgxvllm
dvl
```

You can also run the package as a module:

```bash
uv run python -m dgx_vllm_launcher qwen36-fp8
```

Inspect all resolved profile defaults without launching Docker:

```bash
uv run dvl --show-defaults
```

## Authentication

All default model repositories are public and ungated. Qwen FP8, Qwen 27B,
Gemma, DiffusionGemma, Nemotron, Ornith, and Mistral use a Hugging Face token
when one is available but can run anonymously. Qwen 35B NVFP4 does not request
token injection by default.

An optional token can be supplied for authenticated download rate limits:

```bash
HF_TOKEN=... uv run dvl qwen36-fp8 --reasoning
```

The launcher checks `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_HOME/token`, `~/.cache/huggingface/token`, and `~/.huggingface/token`. The `HUGGING_FACE_HUB_TOKEN` alias and `~/.huggingface/token` fallback are deprecated through the 0.1.x releases and targeted for removal in v0.2.0. Tokens are passed through the Docker child-process environment and are never embedded in the Docker command line. The host token file is not mounted into the container; only the Hugging Face `hub` and `xet` cache subdirectories are persisted.

## Command-line usage

```text
dvl [variant] [-r|--reasoning] [-w|--no-warmup]
              [-s|--no-smoke-check] [-d|--detach]
              [-m|--moe-backend <name>] [-l|--linear-backend <name>]
              [-R|--restart-policy <policy>]
              [--use-preloaded-models]
              [--preloaded-models-dir <path>]
              [--show-defaults]
```

Arguments:

- `-r, --reasoning` — enable the selected profile's reasoning path; for DiffusionGemma and Nemotron Omni this makes thinking the server default
- `-w, --no-warmup` — skip post-health warmup requests
- `-s, --no-smoke-check` — skip the final completion smoke check
- `-d, --detach` — exit only after all enabled startup checks pass
- `-m, --moe-backend <name>` — override the profile's vLLM MoE backend
- `-l, --linear-backend <name>` — override the profile's vLLM linear backend
- `-R, --restart-policy <policy>` — one of `no`, `always`, `unless-stopped`, `on-failure`, or `on-failure:<retries>`
- `--use-preloaded-models` — use the profile's local checkpoint when present; otherwise warn and use Hugging Face
- `--preloaded-models-dir <path>` — override `VLLM_PRELOADED_MODELS_DIR` or the default `~/models`
- `--show-defaults` — print every configured profile and exit; no variant is required

Prefix caching is profile-driven. It remains enabled by default. Native MTP
temporarily disables it pending [vLLM PR #47861](https://github.com/vllm-project/vllm/pull/47861),
while experimental DFlash disables it independently pending GB10 revalidation
of [vLLM issue #42084](https://github.com/vllm-project/vllm/issues/42084).

Reasoning configuration is profile-driven:

- Qwen and Ornith use the Qwen reasoning/tool parsers.
- Gemma uses the Gemma 4 parser, tool parser, and vLLM tool chat template.
- Mistral uses its native tokenizer plus the Mistral reasoning and tool parsers. Clients opt into thinking per request with `reasoning_effort="high"`.
- DiffusionGemma always loads the Gemma 4 reasoning/tool parsers so channel markers never leak into normal text. Thinking defaults off, or on with `--reasoning`; clients can override it per request with `chat_template_kwargs={"enable_thinking": ...}`.
- Nemotron 3 Nano Omni always loads the `nemotron_v3` reasoning parser and Qwen 3 Coder tool parser. Thinking defaults off, or on with `--reasoning`; clients can override it per request with `chat_template_kwargs={"enable_thinking": ...}`.

## Preloaded checkpoints

Hosted models remain the default. To prefer a local checkpoint:

```bash
uv run dvl qwen36-nvfp4 --use-preloaded-models
uv run dvl qwen36-27b-nvfp4 --use-preloaded-models
uv run dvl gemma4-nvfp4 --use-preloaded-models
uv run dvl ornith-nvfp4 --use-preloaded-models
uv run dvl mistral4-nvfp4 --use-preloaded-models
uv run dvl diffusion-gemma-nvfp4 --use-preloaded-models
uv run dvl nemotron3-nano-omni-nvfp4 --use-preloaded-models
```

The default root is `~/models`. Override it with either:

```bash
VLLM_PRELOADED_MODELS_DIR=/opt/models uv run dvl gemma4-nvfp4 --use-preloaded-models
uv run dvl gemma4-nvfp4 --use-preloaded-models --preloaded-models-dir /opt/models
```

When the expected directory exists, it is mounted read-only at `/model`. If it is missing, the launcher emits a warning and uses the configured Hugging Face model ID. A selected preloaded Qwen 27B, Gemma, DiffusionGemma, Nemotron, Ornith, or Mistral model does not receive an optional HF token.

## Startup and cleanup behavior

Before replacing a service, the launcher:

1. Resolves and validates the complete launch plan.
2. Validates required credentials and selected local model paths.
3. Creates writable cache and artifact directories.
4. Verifies the Docker daemon and pulls a missing image.
5. Checks that an existing same-name container is managed by this launcher.

New containers carry a versioned management marker and a per-launch identity label. Cleanup verifies that identity before stopping or removing a container. For migration, replacement accepts legacy `managed=true` containers, while the new marker is deliberately unrecognized by launchers using the legacy name-based cleanup logic, so they cannot reap a replacement. A same-name container with neither the current nor legacy launcher marker is never removed automatically; rename or remove it explicitly.

After startup:

1. `/health` must become ready before `VLLM_READY_TIMEOUT` expires.
2. Every enabled warmup request must succeed.
3. The enabled smoke request must succeed.

A failure returns a nonzero status and removes the newly started container. In foreground mode, the container is also stopped and removed when log streaming ends or the launcher is interrupted.

## Detached service mode

```bash
uv run dvl qwen36-nvfp4 \
  --reasoning \
  --detach \
  --restart-policy unless-stopped
```

Once all checks pass, manage the service with Docker:

```bash
docker ps -f name=vllm-qwen36-nvfp4
docker logs -f vllm-qwen36-nvfp4
docker stop vllm-qwen36-nvfp4
```

## Environment variables

### General

- `VLLM_READY_TIMEOUT` — positive readiness timeout in seconds; default `10800` (3 hours)
- `VLLM_WARMUP_REQUESTS` — nonnegative warmup count; default `2`
- `VLLM_BIND_ADDRESS` — IP address on which Docker publishes the API; default `127.0.0.1`. [Docker Engine 28 or newer](https://docs.docker.com/engine/network/port-publishing/) enforces host-only access for localhost publishing. On older engines, same-L2 hosts may still reach a localhost-published port; the launcher warns, so upgrade Docker or enforce equivalent firewall rules. Set `0.0.0.0` or `::` only for deliberate network exposure.
- `VLLM_HOST_PORT` — host port mapped to container port 8000; default `8000`
- `VLLM_CACHE_DIR` — host vLLM/TorchInductor cache; default `~/.cache/vllm`
- `VLLM_HF_CACHE_DIR` — Hugging Face cache root; its `hub` and `xet` subdirectories are persisted, while credentials remain on the host. Defaults to nonempty `HF_HOME` or `~/.cache/huggingface`.
- `VLLM_ARTIFACT_DIR` — warmup and smoke response output; default `/tmp`
- `VLLM_PRELOADED_MODELS_DIR` — preloaded checkpoint root; default `~/models`

### Images

- `VLLM_IMAGE_QWEN36_FP8` — Qwen FP8 image override
- `VLLM_IMAGE_QWEN36_NVFP4` — Qwen 35B NVFP4 image override
- `VLLM_IMAGE_QWEN36_27B_NVFP4` — Qwen 27B native-MTP image override
- `VLLM_IMAGE_QWEN36_27B_NVFP4_DFLASH` — Qwen 27B DFlash image override
- `VLLM_IMAGE_GEMMA4_NVFP4` — Gemma NVFP4 image override
- `VLLM_IMAGE_ORNITH_NVFP4` — Ornith NVFP4 image override
- `VLLM_IMAGE_MISTRAL4_NVFP4` — Mistral Small 4 NVFP4 image override
- `VLLM_IMAGE_DIFFUSION_GEMMA_NVFP4` — DiffusionGemma NVFP4 image override
- `VLLM_IMAGE_NEMOTRON3_NANO_OMNI_NVFP4` — Nemotron 3 Nano Omni NVFP4 image override

Every profile uses an immutable vLLM image digest pinned in
`dgx_vllm_launcher/config.py`. Most share the validated default image; the Qwen
27B DFlash profile uses a newer, separately pinned image containing hybrid
sliding/full-attention DFlash support.
The legacy Qwen overrides `VLLM_IMAGE_FP8` and `VLLM_IMAGE_NVFP4` remain accepted through v0.1.x; the canonical names above take precedence when both are set.

### vLLM and container tuning

- `VLLM_SAFETENSORS_LOAD_STRATEGY` — default `lazy`; applies to profiles using vLLM's standard safetensors loader
- `VLLM_MARLIN_USE_ATOMIC_ADD` — `0` or `1`; default `1`
- `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` — `0` or `1`; default `1`

Use `--linear-backend` and `--moe-backend` for explicit kernel selection. Qwen NVFP4, DiffusionGemma, and Nemotron Omni use `fastsafetensors`, so the standard safetensors strategy does not apply to those profiles.

### Hugging Face

- `HF_TOKEN`
- `HUGGING_FACE_HUB_TOKEN`
- `HF_HOME`

## DGX Spark profile notes

Qwen FP8 and Ornith use conservative single-Spark scheduling defaults:

- `--gpu-memory-utilization 0.7`
- `--max-model-len 131072`
- `--max-num-seqs 4`
- `--max-num-batched-tokens 8192`

Qwen FP8 disables unsupported DeepGEMM paths on GB10, uses Triton MoE, and enables its two-token `mtp` speculative decoder. It also mounts a TP=1/GB10 Triton MoE configuration tuned against the pinned vLLM image and Triton 3.6. The map uses tuned tiles for decode-sized batches and retains vLLM defaults for prefill-sized batches: a fully tuned map improved isolated kernels but regressed real prefill by 4–8%. After shape warmup, repeated server A/B runs improved C4 decode from 156.6–157.1 to 165.9 aggregate tok/s (about 5.7–5.9%); single-stream, C16, 8K/64K prefill, and concurrent long prefill remained within roughly 1%.

The tuned file is hardware-, shape-, and runtime-specific. Its filename limits automatic selection to `NVIDIA_GB10`, E=256/N=512, FP8 block-128 experts; re-tune and revalidate it when changing the image, Triton version, tensor-parallel size, or checkpoint architecture.

Qwen 35B NVFP4 follows NVIDIA's DGX Spark recipe while retaining the launcher's 128K context limit:

- NVIDIA's `nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint with vLLM's `modelopt_fp4` quantizer
- `--gpu-memory-utilization 0.4`
- `--max-num-seqs 4` and `--max-num-batched-tokens 8192`
- FP8 KV cache and FlashInfer attention
- Marlin MoE, required by the checkpoint's W4A16 NVFP4 experts
- Three-token MTP speculative decoding with a Triton drafter
- `fastsafetensors` loading

A separate GB10 tune of the NVFP4 profile's BF16 Triton MTP-drafter MoE kernel was not retained. Although isolated kernels improved 2–8%, the complete server regressed C4 decode and prefill; the target model's Marlin experts were unaffected. The drafter therefore continues to use vLLM's default Triton configuration.

Qwen 35B NVFP4, Gemma 4, and DiffusionGemma use vLLM's `modelopt_fp4` quantizer. Nemotron Omni uses `modelopt_mixed` because its routed experts are NVFP4 while Mamba, attention, and shared-expert layers use FP8. Ornith and Mistral use their checkpoints' declared `compressed-tensors` quantization format and leave MoE backend selection on automatic. Qwen 27B omits an explicit quantizer so vLLM can auto-detect the checkpoint's mixed FP8/NVFP4 ModelOpt configuration.

### Qwen 3.6 27B dense notes

`qwen36-27b-nvfp4` serves NVIDIA's mixed-precision dense checkpoint on one DGX
Spark. vLLM auto-detects its ModelOpt format: transformer MLP linears use NVFP4,
attention and Gated DeltaNet linears use FP8, and the native MTP head remains
BF16. The profile keeps multimodal input enabled, uses a 128K context limit,
automatic KV-cache selection, chunked prefill, and two native MTP speculative
steps. The model contains one MTP layer, so the second step reruns it. A matched
GB10 comparison retained two steps because they improved both single-request
and concurrency-four end-to-end output throughput; a third step added less than
2% and did not justify its lower acceptance.

Matched GB10 measurements on the pinned image, with prefix caching disabled and
256 deterministic output tokens per request:

| Proposer | 2,315-token prompt, C1 | 9,227-token prompt, C4 | Draft acceptance |
| --- | ---: | ---: | ---: |
| None | 11.09 tok/s | 18.12 tok/s | — |
| MTP1 | 16.60 tok/s | 20.92 tok/s | 85.8% |
| MTP2 | 21.03 tok/s | 21.57 tok/s | 73.3% |
| MTP3 | 21.34 tok/s | 21.66 tok/s | about 58% |

MTP2 improves C1 end-to-end output throughput by about 90% and C4 aggregate
output throughput by about 19% over no speculation in this workload. Mean TTFT
increased from 2.07s to 2.22s at C1 and from 27.10s to 29.92s at C4. Automatic
tool choice and exact recovery-code retrieval at 18,473 prompt tokens passed.
Forced `tool_choice: "required"` is not validated on this image: it reproduced
[the upstream speculative-decoding/grammar failure](https://github.com/vllm-project/vllm/issues/46249),
while `tool_choice: "auto"` worked correctly. A native image-chat smoke test
also identified a generated solid-red image correctly.

`qwen36-27b-nvfp4-dflash` is experimental. It pairs the same target with the
public [`z-lab/Qwen3.6-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)
drafter, which the publisher documents against the BF16 Qwen target rather than
NVIDIA's quantized target and describes as still under development. The profile
is intended for comparison and validation, not as the default 27B launch path.
It uses a July 14 vLLM nightly with Model Runner V2, FlashAttention, BF16 KV
cache, five speculative tokens, a 64K context limit, text-only mode, and eager
execution. A `0.50` memory target measured 12.90 GiB of KV cache, or 154,219
tokens and 2.35 full 64K requests. Its 8,208-token scheduler budget leaves 8,192
target-token slots after vLLM reserves the K5 draft slots.

The DFlash K search used the same prompts and output lengths as the native-MTP
comparison:

| Proposer | 2,315-token prompt, C1 | 9,227-token prompt, C4 | Mean TTFT, C1 / C4 | Draft acceptance |
| --- | ---: | ---: | ---: | ---: |
| Native MTP2 | 21.03 tok/s | 21.57 tok/s | 2.22s / 29.92s | 73.3% |
| DFlash K3 | 17.33 tok/s | 18.94 tok/s | 2.61s / 32.78s | 50.7% |
| DFlash K5 | 22.40 tok/s | 18.80 tok/s | 2.62s / 32.97s | 39.4% |

K5 is retained because it was 29% faster than K3 at C1 for less than a 1% C4
loss. It beat native MTP2 by about 7% at C1, but was 13% slower at C4; native
MTP2 is therefore the better general default. DFlash automatic tool choice and
exact recovery-code retrieval at 18,473 prompt tokens passed. Its reasoning
parser works, but this Model Runner V2 image does not support the per-request
`thinking_token_budget` parameter.

Native MTP keeps prefix caching disabled pending
[vLLM PR #47861](https://github.com/vllm-project/vllm/pull/47861). DFlash keeps
it disabled separately: landing that MTP fix is not sufficient to re-enable the
DFlash path without retesting
[vLLM issue #42084](https://github.com/vllm-project/vllm/issues/42084) on GB10.
Eager execution should remain enabled until both the SM120/SM121
[piecewise-CUDA-graph fix](https://github.com/vllm-project/vllm/pull/46324) and
the [NVFP4+DFlash compilation crash](https://github.com/vllm-project/vllm/issues/48234)
are resolved in the pinned image and sustained GB10 testing passes.

MTP and DFlash are alternative vLLM proposer methods and cannot be stacked in
one engine. Use the two variants to compare them. Do not raise DFlash to its
published 15-token setting on Spark without retesting: an upstream Spark report
found K=5 stable while K>=10 combined with prefix caching crashed.

## Mistral Small 4 notes

`mistral4-nvfp4` is tuned for one 128 GB DGX Spark rather than the model card's two-GPU recipe. The 119B MoE checkpoint activates about 6.5B parameters per token and occupies 66.1 GiB after loading. Its profile uses:

- Native Mistral config, tokenizer, and consolidated-safetensors loading
- `compressed-tensors` NVFP4 quantization
- Triton MLA attention, required on GB10/SM121 by the pinned vLLM image
- Automatic FlashInfer CUTLASS linear and MoE kernels
- A 128K context limit, 128 sequences, and 16K batched tokens
- A measured 14 GiB KV cache, providing about 652K cache tokens or 4.98 full 128K contexts
- Chunked prefill, prefix caching, four images per prompt, and automatic asynchronous scheduling

The fixed KV budget and `--skip-mm-profiling` avoid a roughly 22-minute synthetic multimodal profile at 128 sequences while retaining measured memory headroom. Validation included eight concurrent prompts with four 1024×1024 images each and a 120,015-token prompt; both completed without an OOM. The pinned image contains `mistral_common 1.11.5`, satisfying the checkpoint's `>=1.11.0` requirement.

On the validated GB10, automatic FlashInfer CUTLASS delivered about 31.1 tok/s single-stream decode, 256.7 aggregate tok/s at concurrency 64, and 361.5 tok/s at concurrency 128. The experimental `flashinfer_b12x` backend was slightly slower at low concurrency and substantially slower at concurrency 16–32, so it is not forced. Mistral's EAGLE draft is also not enabled: the only MLA decode backend supported on SM121 in this image is Triton MLA, whose single-query path does not support speculative decoding.

Use `--reasoning` to install the server-side parser, then send `reasoning_effort="high"` (temperature `0.7` is recommended by Mistral) on requests that should reason. Tool calls and image input use the normal OpenAI-compatible chat endpoint.

## DiffusionGemma notes

`diffusion-gemma-nvfp4` serves NVIDIA's 18.9 GB ModelOpt checkpoint on one DGX Spark. DiffusionGemma has 25.2B total and 3.8B active parameters, but unlike an autoregressive model it repeatedly denoises a 256-token canvas and commits a completed block. The profile uses:

- Model Runner V2, enabled only inside this profile
- `modelopt_fp4` quantization and `fastsafetensors` loading
- Triton attention for mixed causal/bidirectional attention
- Automatic FlashInfer CUTLASS NVFP4 MoE selection on GB10
- A 256K context limit, four sequences, and 8K batched tokens
- Prefix caching, chunked prefill, up to four images, and one video per request
- Gemma 4 reasoning and tool parsers in both modes
- Thinking off by default and on by default with `--reasoning`

Concurrency is deliberately capped at four. Diffusion state scales with `max_num_seqs × canvas_length × vocabulary_size`; raising it can cause an OOM even though the model and KV cache otherwise have substantial headroom. Do not compare its streaming TPOT directly with an autoregressive model: a client may receive no text while a canvas is denoised and then receive many tokens together.

On the validated GB10, loading occupied 18.22 GiB. At `--gpu-memory-utilization 0.8`, vLLM allocated 59.3 GiB for about 4.60M FP8 KV-cache tokens and left about 29 GiB of host memory available. The checkpoint selects FP8 KV automatically; vLLM reports unit fallback scales, so accuracy-sensitive deployments should retain the autoregressive Gemma 4 profile as a comparison baseline.

Validation covered non-thinking text, parsed reasoning, a complete tool-call round trip, image and video data URLs, four concurrent image requests, and a 240,051-token retrieval prompt. The long prompt returned the exact key but required 470 seconds of prefill. Exact-length raw-completion tests were highly prompt dependent: C1 averaged approximately 54–63 tok/s and C4 approximately 76–90 aggregate tok/s for 256–1024 requested tokens. Short real chat requests completed in 0.4–2.1 seconds, illustrating why one fixed diffusion tok/s number is misleading.

Trade-offs:

- The first visible block has higher latency and streaming is bursty.
- Forced long output can take the full 48 denoising steps per canvas and be slower than favorable benchmark prompts.
- Overall reasoning, coding, vision, and retrieval quality is lower than autoregressive Gemma 4 on Google's published evaluations.
- Tool calls work best with thinking enabled.
- With thinking disabled, the model can still emit a thought channel; vLLM then returns the answer in `reasoning` and may leave `content` null. Clients should consume both fields or enable thinking for those prompts.
- Even a two-token launcher warmup evaluates a full canvas internally, so startup checks are heavier than for other profiles.
- The checkpoint is Apache 2.0 but remains subject to Gemma's terms and prohibited-use policy.

## Nemotron 3 Nano Omni notes

`nemotron3-nano-omni-nvfp4` serves NVIDIA's 20.9 GB mixed-precision checkpoint on one DGX Spark. It accepts text, image, video, and audio inputs and produces text, parsed reasoning, JSON, and tool calls. The profile starts from NVIDIA's single-Spark recipe and uses a GB10-validated memory budget with:

- `modelopt_mixed` quantization: NVFP4 routed experts plus FP8 Mamba, attention, and shared-expert layers
- `--gpu-memory-utilization 0.4`, an FP8 KV cache, a 128K context limit, eight sequences, and 32K batched tokens
- an FP32 Mamba SSM state cache, as required by NVIDIA for accuracy, plus a GB10-tuned selective-state-update map
- Efficient Video Sampling at a 0.5 pruning rate
- one image, one video, and one audio input per prompt
- video sampling at 2 FPS with at most 256 frames
- prefix caching, chunked prefill, `nemotron_v3` reasoning parsing, and Qwen 3 Coder tool parsing
- thinking off by default and on by default with `--reasoning`

The server keeps vLLM generation defaults rather than silently importing checkpoint sampling values. Clients should send NVIDIA's recommended `temperature=0.2`, `top_p=0.95`, and `top_k=1` without thinking, or `temperature=0.6`, `top_p=0.95`, and `top_k=20` with thinking.

The pinned vLLM image omits optional audio wheels. Before starting vLLM, this profile installs exact versions of `av`, `scipy`, `soundfile`, and `soxr` without resolving transitive dependencies. The installed package set is content-keyed by image and package versions under `VLLM_CACHE_DIR/python-packages`, so subsequent starts reuse it without downloading again. Other profiles retain the image's normal entrypoint and do not run this setup step.

For security, the profile deliberately does not set `--allowed-local-media-path`. Send HTTPS or data URLs for media rather than exposing container-local files to API clients.

On the validated GB10, automatic selection used FlashInfer CUTLASS for NVFP4 MoE, FlashInfer scaled matrix multiplication for FP8 linears, and FlashInfer attention. NVIDIA's 0.8 memory setting reserved about 94.0 GiB and left only 16 GiB of shared memory available despite the eight-sequence cap. The 0.4 profile with FP32 SSM state reserves about 40 GiB, leaves about 73 GiB available at initial idle, and still provides roughly 4.4M cache tokens—more than 33 full 128K contexts for at most eight scheduled requests.

The checkpoint declares FP8 KV-cache quantization but contains no `q_scale`, `k_scale`, `v_scale`, or `prob_scale` tensors, so vLLM emits its generic unit-scale accuracy warnings. Do not add `--calculate-kv-scales`: vLLM disables runtime scale calculation for hybrid recurrent models because the profiling pass can produce corrupted scales. NVIDIA explicitly serves and evaluates this NVFP4 checkpoint with FP8 KV cache. In a matched GB10 A/B, FP8 and BF16 KV produced identical answers on five short prompts and exact-key retrieval at 8K, 32K, and 64K. FP8 provided about 4.5M versus 2.44M cache tokens; at 32K/64K it reduced TTFT by 11.4%/8.5% and increased decode throughput by 3.5%/5.2%. At 512 tokens the difference was below 1%, and at 8K FP8 decode was 1.1% faster while TTFT was 3.5% slower. Keep FP8 unless a workload-specific quality evaluation demonstrates a regression.

The remote-code Omni architecture does not trigger vLLM's native NemotronH FP32 auto-hook. Letting the SSM dtype default to FP16 is therefore incorrect: NVIDIA's report keeps Mamba state in FP32, and vLLM documents FP32 as the only default known to avoid NemotronH accuracy issues. FP32 reduces cache capacity from about 5.1M to about 4.4M tokens versus FP16, but the remaining 33-plus full-context slots still far exceed the eight-sequence scheduler cap. The bundled SSU map was reference-validated at all eight supported batch sizes. Its robust kernel speedups over vLLM's GB10 fallback are 18.7%, 13.5%, and 11.7% at batches one, two, and three; noisier gains below 3% at batches four through eight retain the safe default launch shape.

The lower memory budget had no measured throughput penalty. Forced 256-token raw completions delivered about 58.5 tok/s single-stream, 160.5 aggregate tok/s at concurrency four, and 244–300 aggregate tok/s across concurrency-eight runs. Eight simultaneous 120,045-token exact-key requests all succeeded in 167.5 seconds, and eight simultaneous two-minute/240-frame videos all returned the correct result in 44.0 seconds while retaining roughly 57 GiB of available memory. Single-request exact-key retrieval succeeded at 64,059 and 120,060 input tokens in 8.7 and 20.1 seconds. Repeating a 32K-token prefix reduced completion latency from 3.65 to 0.27 seconds, validating Mamba-aligned prefix caching. Text, parsed reasoning, structured JSON, tool calls, image, audio, silent video, and combined video-with-audio data URLs all completed successfully.

## Gemma 4 notes

Gemma keeps multimodal image input enabled while limiting vLLM multimodal profiling. Its profile uses:

- `--gpu-memory-utilization 0.8`
- `--max-num-seqs 32`
- `--max-num-batched-tokens 16384`
- BF16 KV cache, avoiding the checkpoint's uncalibrated FP8 q/prob-scale warnings
- Gemma 4 reasoning/tool parsers and chat template when `--reasoning` is set

On the validated GB10 image, automatic MoE selection uses FlashInfer CUTLASS. It provides the best mixed-workload result: Marlin is about 0.6% faster for single-stream decode but 5–10% slower for long prefill, while vLLM CUTLASS is slower for decode. Keep backend selection automatic unless testing a different image.

The BF16/0.8 profile retains about 19 GiB host-memory headroom at idle and 16 GiB during C16/long-prefill stress, with capacity for roughly 12.7 simultaneous full 128K contexts. Compared with FP8 KV, BF16 costs less than 1% single-stream decode and about 2% at C4, removes the FP8 scale warnings, and was faster in the validated 64K prefill workload. The checkpoint still emits a fused-expert w1/w3 scale warning; runtime flags cannot repair checkpoint scales, so accuracy-sensitive workloads still require evaluation.

Cold starts can spend tens of minutes downloading and loading weights. The validated Mistral download alone took about 100 minutes on the test link, so the default readiness deadline is three hours. Increase it further on slower links if needed:

```bash
VLLM_READY_TIMEOUT=14400 uv run dvl mistral4-nvfp4
```

## Development

Install dependencies and run every check:

```bash
uv sync --group dev
uv run check
```

Individual commands:

```bash
uv run ruff check dgx_vllm_launcher tests
uv run pyright
uv run pytest -q
```
