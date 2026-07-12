from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Variant = Literal[
    "qwen36-fp8",
    "qwen36-nvfp4",
    "gemma4-nvfp4",
    "ornith-nvfp4",
    "mistral4-nvfp4",
    "diffusion-gemma-nvfp4",
    "nemotron3-nano-omni-nvfp4",
]
TokenPolicy = Literal["none", "optional", "required"]

MODEL_BASE = "Qwen/Qwen3.6-35B-A3B"
QWEN_NVFP4_HF_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
GEMMA4_MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"
ORNITH_MODEL = "sakamakismile/Ornith-1.0-35B-NVFP4"
MISTRAL4_MODEL = "mistralai/Mistral-Small-4-119B-2603-NVFP4"
DIFFUSION_GEMMA_MODEL = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
NEMOTRON3_NANO_OMNI_MODEL = (
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
)

QWEN_LOCAL_NVFP4_PATH = "Qwen3.6-35B-A3B-NVFP4"
GEMMA4_LOCAL_NVFP4_PATH = "Gemma-4-26B-A4B-NVFP4"
ORNITH_LOCAL_NVFP4_PATH = "Ornith-1.0-35B-NVFP4"
MISTRAL4_LOCAL_NVFP4_PATH = "Mistral-Small-4-119B-2603-NVFP4"
DIFFUSION_GEMMA_LOCAL_NVFP4_PATH = "diffusiongemma-26B-A4B-it-NVFP4"
NEMOTRON3_NANO_OMNI_LOCAL_NVFP4_PATH = (
    "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
)

DEFAULT_VLLM_IMAGE = "vllm/vllm-openai@sha256:7feb2a09304e3b2d38e224a100316e84fe3205faa7605060609e2c02179cbca6"
DEFAULT_FP8_IMAGE = DEFAULT_VLLM_IMAGE
DEFAULT_NVFP4_IMAGE = DEFAULT_VLLM_IMAGE
DEFAULT_GEMMA4_NVFP4_IMAGE = DEFAULT_VLLM_IMAGE
DEFAULT_ORNITH_NVFP4_IMAGE = DEFAULT_VLLM_IMAGE
DEFAULT_MISTRAL4_NVFP4_IMAGE = DEFAULT_VLLM_IMAGE
DEFAULT_DIFFUSION_GEMMA_NVFP4_IMAGE = DEFAULT_VLLM_IMAGE
DEFAULT_NEMOTRON3_NANO_OMNI_NVFP4_IMAGE = DEFAULT_VLLM_IMAGE

DEFAULT_READY_TIMEOUT = 10800
DEFAULT_VLLM_CACHE_DIR = "~/.cache/vllm"
DEFAULT_HF_CACHE_DIR = "~/.cache/huggingface"
DEFAULT_ARTIFACT_DIR = "/tmp"
# GB10-measured budget paired with --skip-mm-profiling below. Pinning it avoids
# a 22-minute synthetic vision profile without consuming the physical headroom.
MISTRAL4_KV_CACHE_BYTES = 14 * 1024**3
DEFAULT_PRELOADED_MODELS_DIR = "~/models"
DEFAULT_HOST_PORT = 8000
CONTAINER_PORT = 8000
DEFAULT_GPU_MEMORY_UTILIZATION = 0.7
DEFAULT_MAX_MODEL_LEN = 131072
DEFAULT_MAX_NUM_SEQS = 4
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192


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
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS
    load_format: str | None = None
    tuned_config_subdir: str | None = None
    container_env: tuple[tuple[str, str], ...] = ()
    always_enable_parsers: bool = False
    extra_vllm_args: tuple[str, ...] = ()
    reasoning_vllm_args: tuple[str, ...] = ()
    non_reasoning_vllm_args: tuple[str, ...] = ()
    startup_python_packages: tuple[str, ...] = ()


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


QWEN_FP8_RUNTIME_DEFAULTS = VariantRuntimeDefaults(
    moe_backend="triton",
    reasoning_parser="qwen3",
    tool_call_parser="qwen3_coder",
    tuned_config_subdir="tuned_configs/qwen36_fp8",
    container_env=(("VLLM_USE_DEEP_GEMM", "0"),),
    extra_vllm_args=(
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":2}',
    ),
)

QWEN_NVFP4_RUNTIME_DEFAULTS = VariantRuntimeDefaults(
    moe_backend="marlin",
    reasoning_parser="qwen3",
    tool_call_parser="qwen3_xml",
    gpu_memory_utilization=0.4,
    load_format="fastsafetensors",
    extra_vllm_args=(
        "--kv-cache-dtype",
        "fp8",
        "--attention-backend",
        "flashinfer",
        "--enable-chunked-prefill",
        "--async-scheduling",
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}',
    ),
)

GEMMA4_RUNTIME_DEFAULTS = VariantRuntimeDefaults(
    # Automatic selection uses FlashInfer CUTLASS on GB10 and outperforms the
    # other supported backends across mixed decode and long-prefill workloads.
    reasoning_parser="gemma4",
    tool_call_parser="gemma4",
    chat_template="/vllm-workspace/examples/tool_chat_template_gemma4.jinja",
    gpu_memory_utilization=0.8,
    max_num_seqs=32,
    max_num_batched_tokens=16384,
    extra_vllm_args=(
        "--kv-cache-dtype",
        "bfloat16",
        "--limit-mm-per-prompt",
        '{"image":4,"video":0}',
        "--mm-processor-kwargs",
        '{"max_soft_tokens":280}',
        "--async-scheduling",
    ),
)

ORNITH_RUNTIME_DEFAULTS = VariantRuntimeDefaults(
    reasoning_parser="qwen3",
    tool_call_parser="qwen3_xml",
)

DIFFUSION_GEMMA_RUNTIME_DEFAULTS = VariantRuntimeDefaults(
    reasoning_parser="gemma4",
    tool_call_parser="gemma4",
    gpu_memory_utilization=0.8,
    max_model_len=262144,
    max_num_seqs=4,
    max_num_batched_tokens=8192,
    load_format="fastsafetensors",
    container_env=(("VLLM_USE_V2_MODEL_RUNNER", "1"),),
    always_enable_parsers=True,
    extra_vllm_args=(
        "--attention-backend",
        "TRITON_ATTN",
        "--diffusion-config",
        '{"canvas_length":256}',
        "--override-generation-config",
        '{"max_new_tokens":null}',
        "--limit-mm-per-prompt",
        '{"image":4,"video":1}',
        "--mm-processor-kwargs",
        '{"max_soft_tokens":280}',
        "--enable-chunked-prefill",
    ),
    reasoning_vllm_args=(
        "--default-chat-template-kwargs",
        '{"enable_thinking":true}',
    ),
    non_reasoning_vllm_args=(
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    ),
)

NEMOTRON3_NANO_OMNI_RUNTIME_DEFAULTS = VariantRuntimeDefaults(
    reasoning_parser="nemotron_v3",
    tool_call_parser="qwen3_coder",
    gpu_memory_utilization=0.4,
    max_model_len=131072,
    max_num_seqs=8,
    max_num_batched_tokens=32768,
    load_format="fastsafetensors",
    tuned_config_subdir="tuned_configs/nemotron3_nano_omni",
    always_enable_parsers=True,
    extra_vllm_args=(
        "--kv-cache-dtype",
        "fp8",
        # The remote-code Omni architecture misses vLLM's NemotronH auto-hook;
        # FP32 is NVIDIA's accuracy-preserving SSM state dtype for this model.
        "--mamba-ssm-cache-dtype",
        "float32",
        "--video-pruning-rate",
        "0.5",
        "--limit-mm-per-prompt",
        '{"video":1,"image":1,"audio":1}',
        "--media-io-kwargs",
        '{"video":{"fps":2,"num_frames":256}}',
        "--enable-chunked-prefill",
    ),
    reasoning_vllm_args=(
        "--default-chat-template-kwargs",
        '{"enable_thinking":true}',
    ),
    non_reasoning_vllm_args=(
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    ),
    startup_python_packages=(
        "av==18.0.0",
        "scipy==1.18.0",
        "soundfile==0.14.0",
        "soxr==1.1.0",
    ),
)


MISTRAL4_RUNTIME_DEFAULTS = VariantRuntimeDefaults(
    reasoning_parser="mistral",
    tool_call_parser="mistral",
    gpu_memory_utilization=0.8,
    max_num_seqs=128,
    max_num_batched_tokens=16384,
    load_format="mistral",
    extra_vllm_args=(
        "--tokenizer-mode",
        "mistral",
        "--config-format",
        "mistral",
        "--attention-backend",
        "TRITON_MLA",
        "--limit-mm-per-prompt",
        '{"image":4}',
        "--skip-mm-profiling",
        "--kv-cache-memory-bytes",
        str(MISTRAL4_KV_CACHE_BYTES),
        "--enable-chunked-prefill",
    ),
)

VARIANT_PROFILES: dict[Variant, VariantProfile] = {
    "qwen36-fp8": VariantProfile(
        variant="qwen36-fp8",
        source=HuggingFaceModel(
            f"{MODEL_BASE}-FP8",
            token_policy="optional",
        ),
        image_env_var="VLLM_IMAGE_FP8",
        default_image=DEFAULT_FP8_IMAGE,
        served_model_name="qwen36-fp8",
        startup_message="Serving Qwen/Qwen3.6-35B-A3B-FP8 from Hugging Face...",
        runtime_defaults=QWEN_FP8_RUNTIME_DEFAULTS,
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
        startup_message="Serving nvidia/Qwen3.6-35B-A3B-NVFP4 from Hugging Face...",
        runtime_defaults=QWEN_NVFP4_RUNTIME_DEFAULTS,
        quantization="modelopt_fp4",
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
        quantization="modelopt_fp4",
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
        runtime_defaults=ORNITH_RUNTIME_DEFAULTS,
        quantization="compressed-tensors",
    ),
    "mistral4-nvfp4": VariantProfile(
        variant="mistral4-nvfp4",
        source=HuggingFaceModel(
            MISTRAL4_MODEL,
            token_policy="optional",
            preloaded=PreloadedModel(MISTRAL4_LOCAL_NVFP4_PATH),
        ),
        image_env_var="VLLM_IMAGE_MISTRAL4_NVFP4",
        default_image=DEFAULT_MISTRAL4_NVFP4_IMAGE,
        served_model_name="mistral4-nvfp4",
        startup_message="Serving Mistral Small 4 119B A6B NVFP4 from Hugging Face...",
        runtime_defaults=MISTRAL4_RUNTIME_DEFAULTS,
        quantization="compressed-tensors",
    ),
    "diffusion-gemma-nvfp4": VariantProfile(
        variant="diffusion-gemma-nvfp4",
        source=HuggingFaceModel(
            DIFFUSION_GEMMA_MODEL,
            token_policy="optional",
            preloaded=PreloadedModel(DIFFUSION_GEMMA_LOCAL_NVFP4_PATH),
        ),
        image_env_var="VLLM_IMAGE_DIFFUSION_GEMMA_NVFP4",
        default_image=DEFAULT_DIFFUSION_GEMMA_NVFP4_IMAGE,
        served_model_name="diffusion-gemma-nvfp4",
        startup_message="Serving DiffusionGemma 26B A4B NVFP4 from Hugging Face...",
        runtime_defaults=DIFFUSION_GEMMA_RUNTIME_DEFAULTS,
        quantization="modelopt_fp4",
    ),
    "nemotron3-nano-omni-nvfp4": VariantProfile(
        variant="nemotron3-nano-omni-nvfp4",
        source=HuggingFaceModel(
            NEMOTRON3_NANO_OMNI_MODEL,
            token_policy="optional",
            preloaded=PreloadedModel(NEMOTRON3_NANO_OMNI_LOCAL_NVFP4_PATH),
        ),
        image_env_var="VLLM_IMAGE_NEMOTRON3_NANO_OMNI_NVFP4",
        default_image=DEFAULT_NEMOTRON3_NANO_OMNI_NVFP4_IMAGE,
        served_model_name="nemotron3-nano-omni-nvfp4",
        startup_message=(
            "Serving Nemotron 3 Nano Omni 30B A3B NVFP4 from Hugging Face..."
        ),
        runtime_defaults=NEMOTRON3_NANO_OMNI_RUNTIME_DEFAULTS,
        quantization="modelopt_mixed",
    ),
}

VARIANTS: tuple[Variant, ...] = tuple(VARIANT_PROFILES)


def resolve_variant_profile(variant: Variant) -> VariantProfile:
    try:
        return VARIANT_PROFILES[variant]
    except KeyError as exc:
        raise ValueError(f"unsupported variant: {variant}") from exc
