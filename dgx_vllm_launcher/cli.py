from __future__ import annotations

import argparse
from dataclasses import dataclass

from .config import Variant


@dataclass(frozen=True)
class LaunchArgs:
    variant: Variant
    reasoning: bool = False
    no_warmup: bool = False
    no_smoke_check: bool = False
    enable_prefix_caching: bool = False
    detach: bool = False
    moe_backend: str | None = None
    linear_backend: str | None = None
    restart_policy: str | None = None


def parse_args(argv: list[str] | None = None) -> LaunchArgs:
    parser = argparse.ArgumentParser(
        description="Unified launcher for qwen/gemma vLLM variants.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "variant",
        choices=["qwen36-fp8", "qwen36-nvfp4", "gemma4-nvfp4"],
        help="qwen36-fp8, qwen36-nvfp4, or gemma4-nvfp4",
    )
    parser.add_argument("-r", "--reasoning", action="store_true", help="Enable Qwen reasoning parser + tool-choice path")
    parser.add_argument("-w", "--no-warmup", action="store_true", help="Skip pre-startup warmup requests")
    parser.add_argument(
        "-s",
        "--no-smoke-check",
        action="store_true",
        help="Skip post-startup smoke check",
    )
    parser.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="Run startup checks then exit, leaving container running",
    )
    parser.add_argument(
        "-p",
        "--enable-prefix-caching",
        action="store_true",
        help="Alias/no-op; prefix caching is enabled by default.",
    )
    parser.add_argument("-m", "--moe-backend", type=str, help="Pass through to vLLM --moe-backend")
    parser.add_argument("-l", "--linear-backend", type=str, help="Pass through to vLLM --linear-backend")
    parser.add_argument(
        "-R",
        "--restart-policy",
        type=str,
        help="Optional Docker restart policy (for example: on-failure, unless-stopped)",
    )

    ns = parser.parse_args(argv)
    return LaunchArgs(
        variant=ns.variant,
        reasoning=ns.reasoning,
        no_warmup=ns.no_warmup,
        no_smoke_check=ns.no_smoke_check,
        enable_prefix_caching=ns.enable_prefix_caching,
        detach=ns.detach,
        moe_backend=ns.moe_backend,
        linear_backend=ns.linear_backend,
        restart_policy=ns.restart_policy,
    )


