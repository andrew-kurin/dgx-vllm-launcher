from __future__ import annotations

import os

from .config import VariantRuntimeDefaults


def build_common_args(
    served_model_name: str,
    reasoning: bool,
    runtime_defaults: VariantRuntimeDefaults,
) -> list[str]:
    """Build vLLM server args that are common across Docker launches.

    Model-specific launch choices belong in ``VariantRuntimeDefaults`` and are
    supplied by the resolved variant profile. This keeps the command builder
    generic and avoids hiding model defaults in orchestration code.
    """
    safetensors_load_strategy = os.environ.get("VLLM_SAFETENSORS_LOAD_STRATEGY", "prefetch")

    args = [
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "131072",
        "--max-num-seqs",
        str(runtime_defaults.max_num_seqs),
        "--max-num-batched-tokens",
        str(runtime_defaults.max_num_batched_tokens),
        "--enable-prefix-caching",
        "--enable-flashinfer-autotune",
        "--safetensors-load-strategy",
        safetensors_load_strategy,
        "--generation-config",
        "vllm",
        "--trust-remote-code",
        "--served-model-name",
        served_model_name,
    ]

    if reasoning:
        if not runtime_defaults.reasoning_parser or not runtime_defaults.tool_call_parser:
            raise RuntimeError("reasoning requested but the variant has no configured reasoning/tool parser")
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
    return args
