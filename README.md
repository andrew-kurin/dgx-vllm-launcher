# dgx-vllm-launcher

A validated Docker launcher for serving supported FP8 and NVFP4 models with vLLM.

## Supported variants

| Variant | Default model source | Optional preloaded checkpoint | Default MoE backend |
| --- | --- | --- | --- |
| `qwen36-fp8` | `Qwen/Qwen3.6-35B-A3B-FP8` | — | `(none)` |
| `qwen36-nvfp4` | `Qwen/Qwen3.6-35B-A3B-NVFP4` | `Qwen3.6-35B-A3B-NVFP4` | `flashinfer_b12x` |
| `gemma4-nvfp4` | `nvidia/Gemma-4-26B-A4B-NVFP4` | `Gemma-4-26B-A4B-NVFP4` | `(none)` |
| `ornith-nvfp4` | `sakamakismile/Ornith-1.0-35B-NVFP4` | `Ornith-1.0-35B-NVFP4` | `(none)` |

The launcher provides:

- A single immutable launch plan resolved before Docker is changed
- Profile-driven model, reasoning, tool-calling, and performance settings
- Optional preloaded checkpoints with explicit Hugging Face fallback warnings
- Docker and image preflight checks
- Health polling against a real monotonic deadline
- Required warmup and smoke checks when enabled
- Safe secret forwarding without putting tokens in Docker command arguments
- Managed-container labels, collision protection, and foreground cleanup

## Quick start

```bash
cd path/to/dgx-vllm-launcher
uv sync --group dev

uv run dvl qwen36-fp8
uv run dvl qwen36-nvfp4
uv run dvl gemma4-nvfp4
uv run dvl ornith-nvfp4
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

`qwen36-fp8` requires a Hugging Face token:

```bash
HF_TOKEN=... uv run dvl qwen36-fp8 --reasoning
```

Gemma and Ornith use a token when one is available but can run anonymously. Qwen NVFP4 does not request token injection by default.

The launcher checks `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_HOME/token`, `~/.cache/huggingface/token`, and `~/.huggingface/token`. Tokens are passed through the Docker child-process environment and are never embedded in the Docker command line.

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

- `-r, --reasoning` — enable the selected profile's reasoning parser and automatic tool choice
- `-w, --no-warmup` — skip post-health warmup requests
- `-s, --no-smoke-check` — skip the final completion smoke check
- `-d, --detach` — exit only after all enabled startup checks pass
- `-m, --moe-backend <name>` — override the profile's vLLM MoE backend
- `-l, --linear-backend <name>` — override the profile's vLLM linear backend
- `-R, --restart-policy <policy>` — one of `no`, `always`, `unless-stopped`, `on-failure`, or `on-failure:<retries>`
- `--use-preloaded-models` — use the profile's local checkpoint when present; otherwise warn and use Hugging Face
- `--preloaded-models-dir <path>` — override `VLLM_PRELOADED_MODELS_DIR` or the default `~/models`
- `--show-defaults` — print every recommended profile and exit; no variant is required

Prefix caching is always enabled.

Reasoning configuration is profile-driven:

- Qwen and Ornith use the Qwen reasoning/tool parsers.
- Gemma uses the Gemma 4 parser, tool parser, and vLLM tool chat template.

## Preloaded checkpoints

Hosted models remain the default. To prefer a local checkpoint:

```bash
uv run dvl qwen36-nvfp4 --use-preloaded-models
uv run dvl gemma4-nvfp4 --use-preloaded-models
uv run dvl ornith-nvfp4 --use-preloaded-models
```

The default root is `~/models`. Override it with either:

```bash
VLLM_PRELOADED_MODELS_DIR=/opt/models uv run dvl gemma4-nvfp4 --use-preloaded-models
uv run dvl gemma4-nvfp4 --use-preloaded-models --preloaded-models-dir /opt/models
```

When the expected directory exists, it is mounted read-only at `/model`. If it is missing, the launcher emits a warning and uses the configured Hugging Face model ID. A selected preloaded Gemma or Ornith model does not receive an optional HF token.

## Startup and cleanup behavior

Before replacing a service, the launcher:

1. Resolves and validates the complete launch plan.
2. Validates required credentials and selected local model paths.
3. Creates writable cache and artifact directories.
4. Verifies the Docker daemon and pulls a missing image.
5. Checks that an existing same-name container is managed by this launcher.

Containers carry a management label. A same-name container without that label is never removed automatically; rename or remove it explicitly. Containers created by older launcher versions do not have the label and therefore require one explicit cleanup.

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

- `VLLM_READY_TIMEOUT` — positive readiness timeout in seconds; default `1800`
- `VLLM_WARMUP_REQUESTS` — nonnegative warmup count; default `2`
- `VLLM_HOST_PORT` — host port mapped to container port 8000; default `8000`
- `VLLM_CACHE_DIR` — host vLLM/TorchInductor cache; default `~/.cache/vllm`
- `VLLM_HF_CACHE_DIR` — persisted Hugging Face cache; defaults to `HF_HOME` or `~/.cache/huggingface`
- `VLLM_ARTIFACT_DIR` — warmup and smoke response output; default `/tmp`
- `VLLM_PRELOADED_MODELS_DIR` — preloaded checkpoint root; default `~/models`

### Images

- `VLLM_IMAGE_FP8` — default `vllm/vllm-openai:nightly`
- `VLLM_IMAGE_NVFP4` — Qwen NVFP4 image override
- `VLLM_IMAGE_GEMMA4_NVFP4` — Gemma NVFP4 image override
- `VLLM_IMAGE_ORNITH_NVFP4` — Ornith NVFP4 image override

The NVFP4 defaults use the digest pinned in `dgx_vllm_launcher/config.py`.

### vLLM and container tuning

- `VLLM_SAFETENSORS_LOAD_STRATEGY` — default `prefetch`
- `VLLM_MARLIN_USE_ATOMIC_ADD` — default `1`
- `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` — default `1`
- `VLLM_NVFP4_GEMM_BACKEND` — optional container-level NVFP4 GEMM backend

### Hugging Face

- `HF_TOKEN`
- `HUGGING_FACE_HUB_TOKEN`
- `HF_HOME`

## Gemma 4 notes

Gemma keeps multimodal image input enabled while limiting vLLM multimodal profiling. Its profile uses:

- `--max-num-seqs 32`
- `--max-num-batched-tokens 16384`
- FP8 KV cache
- Gemma 4 reasoning/tool parsers and chat template when `--reasoning` is set

FlashInfer-B12X is intentionally not the default because several vLLM releases do not support Gemma's `GELU_TANH` MoE activation. If needed, benchmark explicit backends on the target machine:

```bash
uv run dvl gemma4-nvfp4 -r -m marlin -l marlin
```

First-run download and compilation can require a longer timeout:

```bash
VLLM_READY_TIMEOUT=3600 uv run dvl gemma4-nvfp4
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
