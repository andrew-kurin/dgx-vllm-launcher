# qwen-vllm-launcher

A lightweight Python launcher for serving Qwen FP8 and NVFP4 models with vLLM.

## What it is

This project provides a single Python entrypoint for running either:

- `fp8` model from Hugging Face
- `nvfp4` model from local path `~/models/Qwen3.6-35B-A3B-NVFP4`

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
uv run qwen-vllm-launcher fp8
uv run qwen-vllm-launcher nvfp4 --moe-backend flashinfer_b12x
```

Or run as a module:

```bash
uv run python -m qwen_vllm_launcher fp8
```

## Command-line usage

```bash
qwen-vllm-launcher <fp8|nvfp4> [-r|--reasoning] [-w|--no-warmup] [-s|--no-smoke-check] [-d|--detach]
                      [-p|--enable-prefix-caching] [-m|--moe-backend <name>] [-l|--linear-backend <name>] [-R|--restart-policy <policy>]
```

- `fp8`
  - serves Hugging Face model `Qwen/Qwen3.6-35B-A3B-FP8`
- `nvfp4`
  - serves local model mounted as `/model` from `~/models/Qwen3.6-35B-A3B-NVFP4`

### Arguments

- `-r, --reasoning`  Enable Qwen reasoning parser + auto tool choice
- `-w, --no-warmup`  Skip startup warmup requests
- `-s, --no-smoke-check`  Skip post-startup smoke check request
- `-d, --detach`  Exit after health/warmup/smoke checks, leaving container running in Docker
- `-p, --enable-prefix-caching`  Alias flag kept for compatibility (prefix caching is enabled by default)
- `-m, --moe-backend <name>`  Pass-through to vLLM `--moe-backend`
- `-l, --linear-backend <name>`  Pass-through to vLLM `--linear-backend`
- `-R, --restart-policy <policy>`  Optional Docker restart policy (`on-failure`, `unless-stopped`, etc.)

### Detached service mode

- `--detach` starts the container, waits for health + warmup + smoke checks, then exits.
- The container keeps running independently with Docker so it survives your shell exiting.
- Use `--restart-policy unless-stopped` or `always` for automatic restarts on machine/daemon restart.

Example:

```bash
qwen-vllm-launcher nvfp4 --reasoning --detach --restart-policy unless-stopped
```

Service management:

```bash
docker ps -f name=vllm-nvfp4
docker logs -f vllm-nvfp4
docker stop vllm-nvfp4
```

## Environment variables

- `HF_TOKEN` (required for `fp8`)
- `VLLM_WARMUP_REQUESTS` (default `2`)
- `VLLM_READY_TIMEOUT_FP8` (default `600`)
- `VLLM_READY_TIMEOUT_NVFP4` (default `1800`)
- `VLLM_CACHE_DIR` (default `~/.cache/vllm`)
- `VLLM_IMAGE_FP8` (default `vllm/vllm-openai:nightly`)
- `VLLM_IMAGE_NVFP4` (default `vllm/vllm-openai@sha256:7feb2a09304e3b2d38e224a100316e84fe3205faa7605060609e2c02179cbca6`)
- `VLLM_SAFETENSORS_LOAD_STRATEGY` (default `prefetch`)
- `VLLM_MARLIN_USE_ATOMIC_ADD` (default `1`)
- `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` (default `1`)

## Notes

- Served model names are fixed for compatibility:
  - `qwen36-fp8`
  - `qwen36-nvfp4`
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
uv run ruff check qwen_vllm_launcher tests
```

- Type check:

```bash
uv run pyright
```

- Tests:

```bash
uv run pytest -q
```
