# dgx-vllm-launcher

A lightweight Python launcher for serving Qwen FP8 / Qwen NVFP4 / Gemma-4 NVFP4 / Ornith 1.0 NVFP4 models with vLLM.

## What it is

This project provides a single Python entrypoint for running either:

- `qwen36-fp8` model from Hugging Face
- `qwen36-nvfp4` model from local path `~/models/Qwen3.6-35B-A3B-NVFP4`
- `gemma4-nvfp4` model from Hugging Face `nvidia/Gemma-4-26B-A4B-NVFP4`
- `ornith-nvfp4` model from Hugging Face `sakamakismile/Ornith-1.0-35B-NVFP4`

It includes:

- Rich-formatted startup output
- Shared argument handling
- Container readiness checks (`/health`)
- Optional startup warmup
- Smoke check request
- Automatic container cleanup on exit (foreground mode)

## Quick start

From repo root:

```bash
cd ~/projects/inference
```

Install dependencies (including dev tools):

```bash
uv sync --group dev
```

Run with Python:

```bash
# Full command
uv run dgx-vllm-launcher qwen36-fp8

# Short command + variant
uv run dvl qwen36-fp8
uv run dvl qwen36-nvfp4
uv run dvl gemma4-nvfp4
uv run dvl ornith-nvfp4

# FP8 needs HuggingFace auth; either env token:
HF_TOKEN=... uv run dvl qwen36-fp8 --reasoning
# or login once with HF CLI (token is auto-detected):
# huggingface-cli login
uv run dvl qwen36-fp8 --reasoning
```
Or run as a module:

```bash
uv run python -m dgx_vllm_launcher qwen36-fp8
```

## Command-line usage

All commands below accept any of:
`dgx-vllm-launcher`, `dgxvllm`, or `dvl`
where the variant is one of:
`qwen36-fp8`, `qwen36-nvfp4`, `gemma4-nvfp4`, or `ornith-nvfp4`.

```bash
dgx-vllm-launcher <qwen36-fp8|qwen36-nvfp4|gemma4-nvfp4|ornith-nvfp4> [-r|--reasoning] [-w|--no-warmup] [-s|--no-smoke-check] [-d|--detach]
                      [-p|--enable-prefix-caching] [-m|--moe-backend <name>] [-l|--linear-backend <name>] [-R|--restart-policy <policy>]
```

- `qwen36-fp8`
  - serves Hugging Face model `Qwen/Qwen3.6-35B-A3B-FP8`
- `qwen36-nvfp4`
  - serves local model mounted as `/model` from `~/models/Qwen3.6-35B-A3B-NVFP4`
- `gemma4-nvfp4`
  - serves Hugging Face model `nvidia/Gemma-4-26B-A4B-NVFP4`
- `ornith-nvfp4`
  - serves Hugging Face model `sakamakismile/Ornith-1.0-35B-NVFP4`

### Arguments

- `-r, --reasoning`  Enable Qwen reasoning parser + auto tool choice (needed for Pi tool calling)
- `-w, --no-warmup`  Skip startup warmup requests
- `-s, --no-smoke-check`  Skip post-startup smoke check request
- `-d, --detach`  Exit after health/warmup/smoke checks, leaving container running in Docker
- `-p, --enable-prefix-caching`  Alias flag kept for compatibility (prefix caching is enabled by default)
- `-m, --moe-backend <name>`  Pass-through to vLLM `--moe-backend` (defaults to `flashinfer_b12x` for `qwen36-nvfp4`, `gemma4-nvfp4`, and `ornith-nvfp4`)
- `-l, --linear-backend <name>`  Pass-through to vLLM `--linear-backend`
- `-R, --restart-policy <policy>`  Optional Docker restart policy (`on-failure`, `unless-stopped`, etc.)

### Detached service mode

- `--detach` starts the container, waits for health + warmup + smoke checks, then exits.
- The container keeps running independently with Docker so it survives your shell exiting.
- Use `--restart-policy unless-stopped` or `always` for automatic restarts on machine/daemon restart.

Example:

```bash
dgx-vllm-launcher qwen36-nvfp4 --reasoning --detach --restart-policy unless-stopped
```

Service management:

```bash
docker ps -f name=vllm-qwen36-nvfp4
docker ps -f name=vllm-gemma4-nvfp4
docker ps -f name=vllm-ornith-nvfp4

docker logs -f vllm-qwen36-nvfp4
# or
docker logs -f vllm-gemma4-nvfp4
# or
docker logs -f vllm-ornith-nvfp4

docker stop vllm-qwen36-nvfp4
# or
docker stop vllm-gemma4-nvfp4
# or
docker stop vllm-ornith-nvfp4
```

## Environment variables

- `HF_TOKEN` (required for `qwen36-fp8`)
- `VLLM_WARMUP_REQUESTS` (default `2`)
- `VLLM_READY_TIMEOUT` (default `1800`) applies to all variants
- `VLLM_CACHE_DIR` (default `~/.cache/vllm`)
- `VLLM_IMAGE_FP8` (default `vllm/vllm-openai:nightly`)
- `VLLM_IMAGE_NVFP4` (default `vllm/vllm-openai@sha256:7feb2a09304e3b2d38e224a100316e84fe3205faa7605060609e2c02179cbca6`)
- `VLLM_IMAGE_GEMMA4_NVFP4` (default same as `VLLM_IMAGE_NVFP4`)
- `VLLM_IMAGE_ORNITH_NVFP4` (default same as `VLLM_IMAGE_NVFP4`)
- `VLLM_SAFETENSORS_LOAD_STRATEGY` (default `prefetch`)
- `VLLM_MARLIN_USE_ATOMIC_ADD` (default `1`)
- `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` (default `1`)

## Notes

- Served model names are fixed for compatibility:
  - `qwen36-fp8`
  - `qwen36-nvfp4`
  - `gemma4-nvfp4`
  - `ornith-nvfp4`
- For NVFP4 startup performance tuning, keep warmup enabled unless you intentionally want to skip it.
- **Pi note:** Pi uses `tool_choice=auto` for tool calling; to support this with vLLM, start with `--reasoning`.

## Developer quickstart

Set up or refresh dependencies:

```bash
uv sync --group dev
```

Run all development checks in one command:

```bash
uv run check
```

Or run individually:

- Lint:

```bash
uv run ruff check dgx_vllm_launcher tests
```

- Type check:

```bash
uv run pyright
```

- Tests:

```bash
uv run pytest -q
```