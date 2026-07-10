from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Variant = Literal["qwen36-fp8", "qwen36-nvfp4", "gemma4-nvfp4", "ornith-nvfp4"]
TokenPolicy = Literal["none", "optional", "required"]

MODEL_BASE = "Qwen/Qwen3.6-35B-A3B"
QWEN_NVFP4_HF_MODEL = "Qwen/Qwen3.6-35B-A3B-NVFP4"
GEMMA4_MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"
ORNITH_MODEL = "sakamakismile/Ornith-1.0-35B-NVFP4"

QWEN_LOCAL_NVFP4_PATH = "Qwen3.6-35B-A3B-NVFP4"
GEMMA4_LOCAL_NVFP4_PATH = "Gemma-4-26B-A4B-NVFP4"
ORNITH_LOCAL_NVFP4_PATH = "Ornith-1.0-35B-NVFP4"

DEFAULT_FP8_IMAGE = "vllm/vllm-openai:nightly"
DEFAULT_NVFP4_IMAGE = "vllm/vllm-openai@sha256:7feb2a09304e3b2d38e224a100316e84fe3205faa7605060609e2c02179cbca6"
DEFAULT_GEMMA4_NVFP4_IMAGE = DEFAULT_NVFP4_IMAGE
DEFAULT_ORNITH_NVFP4_IMAGE = DEFAULT_NVFP4_IMAGE

DEFAULT_READY_TIMEOUT = 1800
DEFAULT_VLLM_CACHE_DIR = "~/.cache/vllm"
DEFAULT_HF_CACHE_DIR = "~/.cache/huggingface"
DEFAULT_ARTIFACT_DIR = "/tmp"
DEFAULT_PRELOADED_MODELS_DIR = "~/models"
DEFAULT_HOST_PORT = 8000
CONTAINER_PORT = 8000
DEFAULT_MAX_NUM_SEQS = 256
DEFAULT_MAX_NUM_BATCHED_TOKENS = 65536


@dataclass(frozen=True)
class PreloadedModel:
    relative_path: str


@dataclass(frozen=True)
class HuggingFaceModel:
    model_id: str
    token_policy: TokenPolicy = "none"
    preloaded: PreloadedModel | None = None


@dataclass(frozen=True)
class VariantRuntimeDefaults:
    """Best-known profile-specific vLLM settings."""

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
    """Static model capabilities and defaults for one launcher variant."""

    variant: Variant
    source: HuggingFaceModel
    image_env_var: str
    default_image: str
    served_model_name: str
    startup_message: str
    runtime_defaults: VariantRuntimeDefaults
    quantization: str | None = None

    @property
    def model(self) -> str:
        return self.source.model_id

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
    # FlashInfer-B12X does not support GELU_TANH in several vLLM releases.
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
        source=HuggingFaceModel(
            f"{MODEL_BASE}-FP8",
            token_policy="required",
        ),
        image_env_var="VLLM_IMAGE_FP8",
        default_image=DEFAULT_FP8_IMAGE,
        served_model_name="qwen36-fp8",
        startup_message="Serving Qwen/Qwen3.6-35B-A3B-FP8 from Hugging Face...",
        runtime_defaults=QWEN_RUNTIME_DEFAULTS,
    ),
    "qwen36-nvfp4": VariantProfile(
        variant="qwen36-nvfp4",
        source=HuggingFaceModel(
            QWEN_NVFP4_HF_MODEL,
            preloaded=PreloadedModel(QWEN_LOCAL_NVFP4_PATH),
        ),
        image_env_var="VLLM_IMAGE_NVFP4",
        default_image=DEFAULT_NVFP4_IMAGE,
        served_model_name="qwen36-nvfp4",
        startup_message="Serving Qwen3.6 NVFP4 from Hugging Face...",
        runtime_defaults=QWEN_NVFP4_RUNTIME_DEFAULTS,
        quantization="modelopt",
    ),
    "gemma4-nvfp4": VariantProfile(
        variant="gemma4-nvfp4",
        source=HuggingFaceModel(
            GEMMA4_MODEL,
            token_policy="optional",
            preloaded=PreloadedModel(GEMMA4_LOCAL_NVFP4_PATH),
        ),
        image_env_var="VLLM_IMAGE_GEMMA4_NVFP4",
        default_image=DEFAULT_GEMMA4_NVFP4_IMAGE,
        served_model_name="gemma4-nvfp4",
        startup_message="Serving Gemma 4 26B A4B-NVFP4 from Hugging Face...",
        runtime_defaults=GEMMA4_RUNTIME_DEFAULTS,
        quantization="modelopt",
    ),
    "ornith-nvfp4": VariantProfile(
        variant="ornith-nvfp4",
        source=HuggingFaceModel(
            ORNITH_MODEL,
            token_policy="optional",
            preloaded=PreloadedModel(ORNITH_LOCAL_NVFP4_PATH),
        ),
        image_env_var="VLLM_IMAGE_ORNITH_NVFP4",
        default_image=DEFAULT_ORNITH_NVFP4_IMAGE,
        served_model_name="ornith-nvfp4",
        startup_message="Serving Ornith 1.0 35B NVFP4 from Hugging Face...",
        runtime_defaults=QWEN_RUNTIME_DEFAULTS,
        quantization="modelopt",
    ),
}

VARIANTS: tuple[Variant, ...] = tuple(VARIANT_PROFILES)


def resolve_variant_profile(variant: Variant) -> VariantProfile:
    try:
        return VARIANT_PROFILES[variant]
    except KeyError as exc:
        raise ValueError(f"unsupported variant: {variant}") from exc
