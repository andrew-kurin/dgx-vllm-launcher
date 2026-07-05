"""Project-level check command: lint + type check + tests."""

from __future__ import annotations

import subprocess
import sys


def _run_python_module(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    subprocess.run(cmd, check=True)


def main() -> int:
    """Run all project checks."""
    checks = [
        ("ruff", ["check", "qwen_vllm_launcher", "tests"]),
        ("pyright", []),
        ("pytest", ["-q"]),
    ]

    for module, args in checks:
        _run_python_module(module, *args)

    return 0
