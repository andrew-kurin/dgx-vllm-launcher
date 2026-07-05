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
uv run dgx-vllm-launcher fp8
# or explicit family-prefixed aliases
uv run dvl-qwen-fp8
uv run dvl-qwen-nvfp4
uv run dvl-gemma4-nvfp4
# or legacy shorthand (still supported)
uv run dvl fp8
uv run dgxvllm nvfp4
```

## Backward-compatibility changes policy

If you keep backward-compatibility support (aliases, fallback env vars, migration shims, deprecated flags, etc.):

- Add a `TODO` comment/issue in code or docs at the point of compatibility logic.
- Include a cleanup plan and (if possible) a target date/version for removal.
- Prefer naming the TODO with the reason, e.g. `TODO(remove backward-compat shim): ...`
