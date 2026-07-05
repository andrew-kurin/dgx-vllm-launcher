from __future__ import annotations

import sys

from .orchestrator import main


def _run_with_variant(variant: str) -> int:
    return main([variant, *sys.argv[1:]])


def qwen_fp8() -> int:
    return _run_with_variant("fp8")


def qwen_nvfp4() -> int:
    return _run_with_variant("nvfp4")


def gemma4_nvfp4() -> int:
    return _run_with_variant("gemma4-nvfp4")
