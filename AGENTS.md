# AGENTS

This repository uses a Python package launcher for vLLM and follows a lightweight local development workflow.

## Single command for all checks

Use this command before committing or after changes:

```bash
uv run check
```

## Developer quickstart

```bash
# Install/deps (including dev tooling)
uv sync --group dev

# Run the launcher
uv run qwen-vllm-launcher fp8
# or
uv run qwen-vllm-launcher nvfp4
```
