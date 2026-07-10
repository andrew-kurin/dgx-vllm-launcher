from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from dgx_vllm_launcher.cli import LaunchArgs
from dgx_vllm_launcher.config import Variant
from dgx_vllm_launcher.plan import LaunchPlan, resolve_launch_plan

PlanFactory = Callable[..., LaunchPlan]


@pytest.fixture
def make_plan(tmp_path: Path) -> PlanFactory:
    preloaded_root = tmp_path / "preloaded-models"
    preloaded_root.mkdir()
    base_env = {
        "VLLM_CACHE_DIR": str(tmp_path / "vllm-cache"),
        "VLLM_HF_CACHE_DIR": str(tmp_path / "hf-cache"),
        "VLLM_ARTIFACT_DIR": str(tmp_path / "artifacts"),
        "VLLM_PRELOADED_MODELS_DIR": str(preloaded_root),
    }

    def factory(
        variant: Variant = "qwen36-nvfp4",
        *,
        env_overrides: Mapping[str, str] | None = None,
        **arg_overrides: Any,
    ) -> LaunchPlan:
        values: dict[str, Any] = {
            "variant": variant,
            "reasoning": False,
            "no_warmup": False,
            "no_smoke_check": False,
            "detach": False,
            "moe_backend": None,
            "linear_backend": None,
            "restart_policy": None,
            "use_preloaded_models": False,
            "preloaded_models_dir": None,
            "show_defaults": False,
        }
        values.update(arg_overrides)
        env = dict(base_env)
        if env_overrides:
            env.update(env_overrides)
        return resolve_launch_plan(LaunchArgs(**values), env)

    return factory
