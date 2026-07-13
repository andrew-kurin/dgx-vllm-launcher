"""Project-level check command: lint + type check + tests."""

from __future__ import annotations

import subprocess
import sys


def _run_python_module(module: str, *args: str) -> int:
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    """Run all project checks."""
    checks = [
        ("ruff", ["check", "dgx_vllm_launcher", "tests"]),
        ("pyright", []),
        ("pytest", ["-q"]),
    ]

    for module, args in checks:
        returncode = _run_python_module(module, *args)
        if returncode != 0:
            return returncode

    return 0
