from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Callable
import os

Variant = Literal["qwen36-fp8", "qwen36-nvfp4", "gemma4-nvfp4"]

MODEL_BASE = "Qwen/Qwen3.6-35B-A3B"
GEMMA4_MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"

DEFAULT_FP8_IMAGE = "vllm/vllm-openai:nightly"
DEFAULT_NVFP4_IMAGE = "vllm/vllm-openai@sha256:7feb2a09304e3b2d38e224a100316e84fe3205faa7605060609e2c02179cbca6"
DEFAULT_GEMMA4_NVFP4_IMAGE = DEFAULT_NVFP4_IMAGE

DEFAULT_READY_TIMEOUT = 1800
DEFAULT_VLLM_CACHE_DIR = "~/.cache/vllm"


@dataclass(frozen=True)
class VariantConfig:
    variant: Variant
    model: str
    image: str
    served_model_name: str
    ready_timeout_seconds: int


@dataclass(frozen=True)
class LaunchConfig:
    variant: Variant
    host_vllm_cache_dir: str
    reasoning: bool
    no_warmup: bool
    moe_backend: str | None
    linear_backend: str | None
    restart_policy: str | None
    variant_config: VariantConfig


def resolve_variant_config(variant: Variant, env_getter: Callable[[str, str], str] = os.getenv) -> VariantConfig:
    timeout = env_getter("VLLM_READY_TIMEOUT", str(DEFAULT_READY_TIMEOUT))

    if variant == "qwen36-fp8":
        model = f"{MODEL_BASE}-FP8"
        image = env_getter("VLLM_IMAGE_FP8", DEFAULT_FP8_IMAGE)
        served = "qwen36-fp8"
    elif variant == "qwen36-nvfp4":
        model = "/model"
        image = env_getter(
            "VLLM_IMAGE_NVFP4",
            DEFAULT_NVFP4_IMAGE,
        )
        served = "qwen36-nvfp4"
    elif variant == "gemma4-nvfp4":
        model = GEMMA4_MODEL
        image = env_getter(
            "VLLM_IMAGE_GEMMA4_NVFP4",
            DEFAULT_GEMMA4_NVFP4_IMAGE,
        )
        served = "gemma4-nvfp4"
    else:
        raise ValueError(f"unsupported variant: {variant}")

    try:
        timeout_seconds = int(timeout)
    except ValueError as exc:
        raise RuntimeError(f"Invalid timeout env value for {variant}: {timeout!r}") from exc

    return VariantConfig(
        variant=variant,
        model=model,
        image=image,
        served_model_name=served,
        ready_timeout_seconds=timeout_seconds,
    )


def resolve_cache_dir(env_getter: Callable[[str, str], str] = os.getenv) -> str:
    return os.path.expanduser(env_getter("VLLM_CACHE_DIR", DEFAULT_VLLM_CACHE_DIR))


def resolve_env_int(name: str, default: int, env_getter: Callable[[str, str], str] = os.getenv) -> int:
    value = env_getter(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer; got {value!r}") from exc
