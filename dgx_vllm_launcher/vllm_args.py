from __future__ import annotations

from .config import VariantRuntimeDefaults


def build_vllm_args(
    served_model_name: str,
    *,
    reasoning: bool,
    runtime_defaults: VariantRuntimeDefaults,
    container_port: int,
    safetensors_load_strategy: str,
    quantization: str | None,
    moe_backend: str | None,
    linear_backend: str | None,
) -> tuple[str, ...]:
    """Build the complete profile-driven vLLM argument list."""

    args = [
        "--host",
        "0.0.0.0",
        "--port",
        str(container_port),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(runtime_defaults.gpu_memory_utilization),
        "--max-model-len",
        str(runtime_defaults.max_model_len),
        "--max-num-seqs",
        str(runtime_defaults.max_num_seqs),
        "--max-num-batched-tokens",
        str(runtime_defaults.max_num_batched_tokens),
        "--enable-flashinfer-autotune",
        "--generation-config",
        "vllm",
        "--trust-remote-code",
        "--served-model-name",
        served_model_name,
    ]

    args.append(
        "--enable-prefix-caching"
        if runtime_defaults.enable_prefix_caching
        else "--no-enable-prefix-caching"
    )

    if runtime_defaults.load_format:
        args.extend(["--load-format", runtime_defaults.load_format])
    else:
        args.extend(["--safetensors-load-strategy", safetensors_load_strategy])

    if quantization:
        args.extend(["--quantization", quantization])
    if moe_backend:
        args.extend(["--moe-backend", moe_backend])
    if linear_backend:
        args.extend(["--linear-backend", linear_backend])

    if reasoning or runtime_defaults.always_enable_parsers:
        if (
            not runtime_defaults.reasoning_parser
            or not runtime_defaults.tool_call_parser
        ):
            raise ValueError(
                "reasoning requested but the variant has no configured reasoning/tool parser"
            )
        args.extend(
            [
                "--reasoning-parser",
                runtime_defaults.reasoning_parser,
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                runtime_defaults.tool_call_parser,
            ]
        )
        if runtime_defaults.chat_template:
            args.extend(["--chat-template", runtime_defaults.chat_template])

    args.extend(runtime_defaults.extra_vllm_args)
    args.extend(
        runtime_defaults.reasoning_vllm_args
        if reasoning
        else runtime_defaults.non_reasoning_vllm_args
    )
    return tuple(args)
