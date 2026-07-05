from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Literal

Variant = Literal["qwen36-fp8", "qwen36-nvfp4", "gemma4-nvfp4", "ornith-nvfp4"]

VARIANTS: tuple[Variant, ...] = (
    "qwen36-fp8",
    "qwen36-nvfp4",
    "gemma4-nvfp4",
    "ornith-nvfp4",
)

MODEL_BASE = "Qwen/Qwen3.6-35B-A3B"
GEMMA4_MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"
ORNITH_MODEL = "sakamakismile/Ornith-1.0-35B-NVFP4"
QWEN_LOCAL_NVFP4_PATH = "Qwen3.6-35B-A3B-NVFP4"
GEMMA4_LOCAL_NVFP4_PATH = "Gemma-4-26B-A4B-NVFP4"
ORNITH_LOCAL_NVFP4_PATH = "Ornith-1.0-35B-NVFP4"

DEFAULT_PRELOADED_MODELS_DIR = "~/models"
QWEN_NVFP4_HF_MODEL = "Qwen/Qwen3.6-35B-A3B-NVFP4"

DEFAULT_FP8_IMAGE = "vllm/vllm-openai:nightly"
DEFAULT_NVFP4_IMAGE = "vllm/vllm-openai@sha256:7feb2a09304e3b2d38e224a100316e84fe3205faa7605060609e2c02179cbca6"
DEFAULT_GEMMA4_NVFP4_IMAGE = DEFAULT_NVFP4_IMAGE
DEFAULT_ORNITH_NVFP4_IMAGE = DEFAULT_NVFP4_IMAGE

DEFAULT_READY_TIMEOUT = 1800
DEFAULT_VLLM_CACHE_DIR = "~/.cache/vllm"
DEFAULT_MAX_NUM_SEQS = 256
DEFAULT_MAX_NUM_BATCHED_TOKENS = 65536


@dataclass(frozen=True)
class VariantRuntimeDefaults:
    """Best-known vLLM defaults for a variant.

    This keeps launch-time behavior explicit and discoverable while still allowing
    every value to be overridden via CLI flags.
    """

    moe_backend: str | None = None
    linear_backend: str | None = None
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    chat_template: str | None = None
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS
    extra_vllm_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class VariantProfile:
    """Static per-variant metadata used by the launcher."""

    variant: Variant
    model: str
    image_env_var: str
    default_image: str
    served_model_name: str
    startup_message: str
    runtime_defaults: VariantRuntimeDefaults
    requires_hf_token: bool
    mount_local_model: bool = False
    local_model_path: str | None = None
    quantization: str | None = None
    inject_hf_token: bool = False

    @property
    def default_moe_backend(self) -> str | None:
        return self.runtime_defaults.moe_backend

    @property
    def default_linear_backend(self) -> str | None:
        return self.runtime_defaults.linear_backend


QWEN_RUNTIME_DEFAULTS = VariantRuntimeDefaults(
    reasoning_parser="qwen3",
    tool_call_parser="qwen3_coder",
)

QWEN_NVFP4_RUNTIME_DEFAULTS = VariantRuntimeDefaults(
    moe_backend="flashinfer_b12x",
    reasoning_parser="qwen3",
    tool_call_parser="qwen3_coder",
)

GEMMA4_RUNTIME_DEFAULTS = VariantRuntimeDefaults(
    # Gemma4 uses GELU_TANH in MoE blocks; the FlashInfer-B12X backend currently does
    # not support this activation in several vLLM releases, so default MoE must be unset.
    reasoning_parser="gemma4",
    tool_call_parser="gemma4",
    chat_template="/vllm-workspace/examples/tool_chat_template_gemma4.jinja",
    max_num_seqs=32,
    max_num_batched_tokens=16384,
    extra_vllm_args=(
        "--kv-cache-dtype",
        "fp8",
        "--limit-mm-per-prompt",
        '{"image":4,"video":0}',
        "--mm-processor-kwargs",
        '{"max_soft_tokens":280}',
        "--async-scheduling",
    ),
)


VARIANT_PROFILES: dict[Variant, VariantProfile] = {
    "qwen36-fp8": VariantProfile(
        variant="qwen36-fp8",
        model=f"{MODEL_BASE}-FP8",
        image_env_var="VLLM_IMAGE_FP8",
        default_image=DEFAULT_FP8_IMAGE,
        served_model_name="qwen36-fp8",
        startup_message="→ Serving Qwen/Qwen3.6-35B-A3B-FP8 from HuggingFace...",
        runtime_defaults=QWEN_RUNTIME_DEFAULTS,
        requires_hf_token=True,
        quantization=None,
        inject_hf_token=True,
    ),
    "qwen36-nvfp4": VariantProfile(
        variant="qwen36-nvfp4",
        model=QWEN_NVFP4_HF_MODEL,
        image_env_var="VLLM_IMAGE_NVFP4",
        default_image=DEFAULT_NVFP4_IMAGE,
        served_model_name="qwen36-nvfp4",
        startup_message="→ Serving Qwen3.6 NVFP4 from Hugging Face...",
        runtime_defaults=QWEN_NVFP4_RUNTIME_DEFAULTS,
        requires_hf_token=False,
        mount_local_model=True,
        local_model_path=QWEN_LOCAL_NVFP4_PATH,
        quantization="modelopt",
        inject_hf_token=False,
    ),
    "gemma4-nvfp4": VariantProfile(
        variant="gemma4-nvfp4",
        model=GEMMA4_MODEL,
        image_env_var="VLLM_IMAGE_GEMMA4_NVFP4",
        default_image=DEFAULT_GEMMA4_NVFP4_IMAGE,
        served_model_name="gemma4-nvfp4",
        startup_message="→ Serving Gemma 4 26B A4B-NVFP4 from Hugging Face...",
        runtime_defaults=GEMMA4_RUNTIME_DEFAULTS,
        requires_hf_token=False,
        local_model_path=GEMMA4_LOCAL_NVFP4_PATH,
        mount_local_model=True,
        quantization="modelopt",
        inject_hf_token=True,
    ),
    "ornith-nvfp4": VariantProfile(
        variant="ornith-nvfp4",
        model=ORNITH_MODEL,
        image_env_var="VLLM_IMAGE_ORNITH_NVFP4",
        default_image=DEFAULT_ORNITH_NVFP4_IMAGE,
        served_model_name="ornith-nvfp4",
        startup_message="→ Serving Ornith 1.0 35B NVFP4 from Hugging Face...",
        runtime_defaults=QWEN_RUNTIME_DEFAULTS,
        requires_hf_token=False,
        local_model_path=ORNITH_LOCAL_NVFP4_PATH,
        mount_local_model=True,
        quantization="modelopt",
        inject_hf_token=True,
    ),
}


@dataclass(frozen=True)
class VariantConfig:
    variant: Variant
    model: str
    image: str
    served_model_name: str
    ready_timeout_seconds: int


def resolve_variant_profile(variant: Variant) -> VariantProfile:
    try:
        return VARIANT_PROFILES[variant]
    except KeyError as exc:
        raise ValueError(f"unsupported variant: {variant}") from exc


def resolve_variant_config(variant: Variant, env_getter: Callable[[str, str], str] = os.getenv) -> VariantConfig:
    timeout = env_getter("VLLM_READY_TIMEOUT", str(DEFAULT_READY_TIMEOUT))
    profile = resolve_variant_profile(variant)

    image = env_getter(profile.image_env_var, profile.default_image)

    try:
        timeout_seconds = int(timeout)
    except ValueError as exc:
        raise RuntimeError(f"Invalid timeout env value for {variant}: {timeout!r}") from exc

    return VariantConfig(
        variant=variant,
        model=profile.model,
        image=image,
        served_model_name=profile.served_model_name,
        ready_timeout_seconds=timeout_seconds,
    )


def resolve_cache_dir(env_getter: Callable[[str, str], str] = os.getenv) -> str:
    return os.path.expanduser(env_getter("VLLM_CACHE_DIR", DEFAULT_VLLM_CACHE_DIR))


def resolve_preloaded_models_root(
    override_root: str | None = None,
    env_getter: Callable[[str, str], str] = os.getenv,
) -> str:
    """Resolve the base directory for local preloaded checkpoints."""
    if override_root is None:
        override_root = env_getter("VLLM_PRELOADED_MODELS_DIR", DEFAULT_PRELOADED_MODELS_DIR)
    return os.path.expanduser(override_root)


def resolve_env_int(name: str, default: int, env_getter: Callable[[str, str], str] = os.getenv) -> int:
    value = env_getter(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer; got {value!r}") from exc
