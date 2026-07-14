from __future__ import annotations

import argparse

from .config import VARIANTS
# TODO(remove backward-compat LaunchArgs re-export): Remove this import from the
# CLI namespace in v0.2.0 after callers migrate to dgx_vllm_launcher.plan.
from .plan import LaunchArgs


def parse_args(argv: list[str] | None = None) -> LaunchArgs:
    parser = argparse.ArgumentParser(
        description=f"Unified launcher for vLLM variants: {', '.join(VARIANTS)}.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "variant",
        choices=list(VARIANTS),
        nargs="?",
        help=", ".join(VARIANTS),
    )
    parser.add_argument(
        "-r",
        "--reasoning",
        action="store_true",
        help=(
            "Enable the selected model's reasoning path; for DiffusionGemma "
            "and Nemotron Omni, make thinking the server default"
        ),
    )
    parser.add_argument(
        "-w",
        "--no-warmup",
        action="store_true",
        help="Skip post-health warmup requests",
    )
    parser.add_argument(
        "-s",
        "--no-smoke-check",
        action="store_true",
        help="Skip the post-warmup smoke check",
    )
    parser.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="Run startup checks then exit, leaving the container running",
    )
    parser.add_argument(
        "-m",
        "--moe-backend",
        type=str,
        help="Pass through to vLLM --moe-backend",
    )
    parser.add_argument(
        "-l",
        "--linear-backend",
        type=str,
        help="Pass through to vLLM --linear-backend",
    )
    parser.add_argument(
        "-R",
        "--restart-policy",
        type=str,
        help="Docker restart policy (for example: on-failure, unless-stopped)",
    )
    parser.add_argument(
        "--use-preloaded-models",
        action="store_true",
        help=(
            "Prefer an available checkpoint under ~/models (or "
            "--preloaded-models-dir); otherwise use Hugging Face"
        ),
    )
    parser.add_argument(
        "--preloaded-models-dir",
        type=str,
        help=(
            "Override the preloaded-model root from VLLM_PRELOADED_MODELS_DIR "
            "or ~/models"
        ),
    )
    parser.add_argument(
        "--show-defaults",
        action="store_true",
        help="Print configured per-variant launch settings and exit",
    )

    ns = parser.parse_args(argv)
    if not ns.show_defaults and ns.variant is None:
        parser.error("the following arguments are required: variant")

    return LaunchArgs(
        variant=ns.variant,
        reasoning=ns.reasoning,
        no_warmup=ns.no_warmup,
        no_smoke_check=ns.no_smoke_check,
        detach=ns.detach,
        moe_backend=ns.moe_backend,
        linear_backend=ns.linear_backend,
        restart_policy=ns.restart_policy,
        use_preloaded_models=ns.use_preloaded_models,
        preloaded_models_dir=ns.preloaded_models_dir,
        show_defaults=ns.show_defaults,
    )
