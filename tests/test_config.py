from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dgx_vllm_launcher.cli import LaunchArgs
from dgx_vllm_launcher.config import (
    DEFAULT_DIFFUSION_GEMMA_NVFP4_IMAGE,
    DEFAULT_FP8_IMAGE,
    DEFAULT_GEMMA4_NVFP4_IMAGE,
    DEFAULT_MISTRAL4_NVFP4_IMAGE,
    DEFAULT_NEMOTRON3_NANO_OMNI_NVFP4_IMAGE,
    DEFAULT_ORNITH_NVFP4_IMAGE,
    DEFAULT_READY_TIMEOUT,
    HuggingFaceModel,
    VARIANTS,
    VARIANT_PROFILES,
)
from dgx_vllm_launcher.plan import ConfigurationError, resolve_launch_plan


def test_variant_profiles_are_complete_and_use_typed_sources():
    assert set(VARIANTS) == set(VARIANT_PROFILES)
    assert all(
        isinstance(profile.source, HuggingFaceModel)
        for profile in VARIANT_PROFILES.values()
    )


def test_default_images_are_pinned_by_digest():
    assert all(
        "@sha256:" in profile.default_image
        for profile in VARIANT_PROFILES.values()
    )


def test_default_ready_timeout_accommodates_cold_mistral_downloads():
    assert DEFAULT_READY_TIMEOUT == 10800


def test_startup_python_packages_must_be_exactly_pinned(monkeypatch):
    profile = VARIANT_PROFILES["nemotron3-nano-omni-nvfp4"]
    monkeypatch.setitem(
        VARIANT_PROFILES,
        "nemotron3-nano-omni-nvfp4",
        replace(
            profile,
            runtime_defaults=replace(
                profile.runtime_defaults,
                startup_python_packages=("av",),
            ),
        ),
    )

    with pytest.raises(ConfigurationError, match="must be exactly pinned"):
        resolve_launch_plan(LaunchArgs(variant="nemotron3-nano-omni-nvfp4"), {})


def test_profiles_preserve_latest_model_and_runtime_defaults():
    fp8 = VARIANT_PROFILES["qwen36-fp8"]
    qwen_nvfp4 = VARIANT_PROFILES["qwen36-nvfp4"]
    gemma = VARIANT_PROFILES["gemma4-nvfp4"]
    ornith = VARIANT_PROFILES["ornith-nvfp4"]
    mistral = VARIANT_PROFILES["mistral4-nvfp4"]
    diffusion_gemma = VARIANT_PROFILES["diffusion-gemma-nvfp4"]
    nemotron_omni = VARIANT_PROFILES["nemotron3-nano-omni-nvfp4"]

    assert fp8.source.token_policy == "optional"
    assert fp8.default_moe_backend == "triton"
    assert fp8.runtime_defaults.container_env == (("VLLM_USE_DEEP_GEMM", "0"),)
    assert fp8.runtime_defaults.tuned_config_subdir == "tuned_configs/qwen36_fp8"
    assert fp8.runtime_defaults.reasoning_parser == "qwen3"
    assert fp8.runtime_defaults.max_num_seqs == 4
    assert fp8.runtime_defaults.max_num_batched_tokens == 8192
    assert qwen_nvfp4.model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert qwen_nvfp4.default_moe_backend == "marlin"
    assert qwen_nvfp4.runtime_defaults.tuned_config_subdir is None
    assert qwen_nvfp4.quantization == "modelopt_fp4"
    assert qwen_nvfp4.runtime_defaults.gpu_memory_utilization == 0.4
    assert qwen_nvfp4.runtime_defaults.load_format == "fastsafetensors"
    assert qwen_nvfp4.source.preloaded is not None
    assert gemma.default_moe_backend is None
    assert gemma.quantization == "modelopt_fp4"
    assert gemma.runtime_defaults.reasoning_parser == "gemma4"
    assert gemma.runtime_defaults.gpu_memory_utilization == 0.8
    assert gemma.runtime_defaults.max_num_seqs == 32
    assert "--limit-mm-per-prompt" in gemma.runtime_defaults.extra_vllm_args
    assert gemma.source.token_policy == "optional"
    assert ornith.default_moe_backend is None
    assert ornith.quantization == "compressed-tensors"
    assert ornith.source.token_policy == "optional"
    assert mistral.model == "mistralai/Mistral-Small-4-119B-2603-NVFP4"
    assert mistral.default_moe_backend is None
    assert mistral.quantization == "compressed-tensors"
    assert mistral.runtime_defaults.reasoning_parser == "mistral"
    assert mistral.source.token_policy == "optional"
    assert mistral.source.preloaded is not None
    assert (
        mistral.source.preloaded.relative_path
        == "Mistral-Small-4-119B-2603-NVFP4"
    )
    assert (
        diffusion_gemma.model == "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
    )
    assert diffusion_gemma.default_moe_backend is None
    assert diffusion_gemma.quantization == "modelopt_fp4"
    assert diffusion_gemma.runtime_defaults.always_enable_parsers is True
    assert diffusion_gemma.source.token_policy == "optional"
    assert diffusion_gemma.source.preloaded is not None
    assert (
        diffusion_gemma.source.preloaded.relative_path
        == "diffusiongemma-26B-A4B-it-NVFP4"
    )
    assert nemotron_omni.model == (
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
    )
    assert nemotron_omni.default_moe_backend is None
    assert nemotron_omni.quantization == "modelopt_mixed"
    assert nemotron_omni.runtime_defaults.reasoning_parser == "nemotron_v3"
    assert nemotron_omni.runtime_defaults.tool_call_parser == "qwen3_coder"
    assert nemotron_omni.runtime_defaults.always_enable_parsers is True
    assert nemotron_omni.runtime_defaults.gpu_memory_utilization == 0.4
    assert nemotron_omni.source.token_policy == "optional"
    assert nemotron_omni.source.preloaded is not None
    assert nemotron_omni.source.preloaded.relative_path == (
        "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
    )


def test_resolve_fp8_plan_defaults():
    plan = resolve_launch_plan(LaunchArgs(variant="qwen36-fp8"), {})

    assert plan.model == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert plan.image == DEFAULT_FP8_IMAGE
    assert plan.served_model_name == "qwen36-fp8"
    assert plan.ready_timeout_seconds == DEFAULT_READY_TIMEOUT
    assert plan.requires_hf_token is False
    assert plan.inject_hf_token is True
    assert plan.base_url == "http://127.0.0.1:8000"


def test_resolve_remote_variant_models_images_and_token_policies():
    gemma = resolve_launch_plan(LaunchArgs(variant="gemma4-nvfp4"), {})
    ornith = resolve_launch_plan(LaunchArgs(variant="ornith-nvfp4"), {})
    mistral = resolve_launch_plan(LaunchArgs(variant="mistral4-nvfp4"), {})
    diffusion_gemma = resolve_launch_plan(
        LaunchArgs(variant="diffusion-gemma-nvfp4"), {}
    )
    nemotron_omni = resolve_launch_plan(
        LaunchArgs(variant="nemotron3-nano-omni-nvfp4"), {}
    )

    assert gemma.model == "nvidia/Gemma-4-26B-A4B-NVFP4"
    assert gemma.image == DEFAULT_GEMMA4_NVFP4_IMAGE
    assert ornith.model == "sakamakismile/Ornith-1.0-35B-NVFP4"
    assert ornith.image == DEFAULT_ORNITH_NVFP4_IMAGE
    assert mistral.model == "mistralai/Mistral-Small-4-119B-2603-NVFP4"
    assert mistral.image == DEFAULT_MISTRAL4_NVFP4_IMAGE
    assert (
        diffusion_gemma.model == "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
    )
    assert diffusion_gemma.image == DEFAULT_DIFFUSION_GEMMA_NVFP4_IMAGE
    assert nemotron_omni.model == (
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
    )
    assert nemotron_omni.image == DEFAULT_NEMOTRON3_NANO_OMNI_NVFP4_IMAGE
    assert gemma.requires_hf_token is False
    assert gemma.inject_hf_token is True
    assert ornith.requires_hf_token is False
    assert ornith.inject_hf_token is True
    assert mistral.requires_hf_token is False
    assert mistral.inject_hf_token is True
    assert diffusion_gemma.requires_hf_token is False
    assert diffusion_gemma.inject_hf_token is True
    assert nemotron_omni.requires_hf_token is False
    assert nemotron_omni.inject_hf_token is True


def test_preloaded_model_is_selected_and_mounted_read_only(
    make_plan,
    tmp_path: Path,
):
    root = tmp_path / "models"
    model_dir = root / "Qwen3.6-35B-A3B-NVFP4"
    model_dir.mkdir(parents=True)

    plan = make_plan(
        use_preloaded_models=True,
        preloaded_models_dir=str(root),
    )

    assert plan.model == "/model"
    assert plan.configured_model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert plan.uses_preloaded_model is True
    assert plan.preloaded_model_path == model_dir
    model_mount = next(
        mount for mount in plan.mounts if mount.container_path == "/model"
    )
    assert model_mount.host_path == model_dir
    assert model_mount.read_only is True


def test_missing_preloaded_model_falls_back_with_warning(make_plan, tmp_path: Path):
    root = tmp_path / "missing-models"

    plan = make_plan(
        use_preloaded_models=True,
        preloaded_models_dir=str(root),
    )

    assert plan.model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert plan.uses_preloaded_model is False
    assert len(plan.warnings) == 1
    assert "Preloaded model not found" in plan.warnings[0]


def test_profile_without_preloaded_candidate_falls_back_with_warning(
    make_plan,
    tmp_path: Path,
):
    plan = make_plan(
        "qwen36-fp8",
        use_preloaded_models=True,
        preloaded_models_dir=str(tmp_path),
    )

    assert plan.model == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert plan.requires_hf_token is False
    assert plan.inject_hf_token is True
    assert "no preloaded checkpoint configured" in plan.warnings[0]


def test_preloaded_gemma_does_not_request_optional_hf_token(
    make_plan,
    tmp_path: Path,
):
    root = tmp_path / "models"
    (root / "Gemma-4-26B-A4B-NVFP4").mkdir(parents=True)

    plan = make_plan(
        "gemma4-nvfp4",
        use_preloaded_models=True,
        preloaded_models_dir=str(root),
    )

    assert plan.uses_preloaded_model is True
    assert plan.inject_hf_token is False
    assert plan.requires_hf_token is False


def test_plan_centralizes_resolved_qwen_arguments(make_plan):
    plan = make_plan(
        moe_backend="custom-moe",
        linear_backend="custom-linear",
        reasoning=True,
    )

    assert _argument_value(plan.vllm_args, "--moe-backend") == "custom-moe"
    assert _argument_value(plan.vllm_args, "--linear-backend") == "custom-linear"
    assert _argument_value(plan.vllm_args, "--quantization") == "modelopt_fp4"
    assert _argument_value(plan.vllm_args, "--reasoning-parser") == "qwen3"
    assert _argument_value(plan.vllm_args, "--tool-call-parser") == "qwen3_xml"


def test_plan_uses_dgx_spark_qwen_nvfp4_arguments(make_plan):
    plan = make_plan("qwen36-nvfp4", reasoning=True)

    assert _argument_value(plan.vllm_args, "--gpu-memory-utilization") == "0.4"
    assert _argument_value(plan.vllm_args, "--max-model-len") == "131072"
    assert _argument_value(plan.vllm_args, "--max-num-seqs") == "4"
    assert _argument_value(plan.vllm_args, "--max-num-batched-tokens") == "8192"
    assert _argument_value(plan.vllm_args, "--load-format") == "fastsafetensors"
    assert "--safetensors-load-strategy" not in plan.vllm_args
    assert _argument_value(plan.vllm_args, "--kv-cache-dtype") == "fp8"
    assert _argument_value(plan.vllm_args, "--attention-backend") == "flashinfer"
    assert _argument_value(plan.vllm_args, "--moe-backend") == "marlin"
    assert _argument_value(plan.vllm_args, "--tool-call-parser") == "qwen3_xml"
    assert _argument_value(plan.vllm_args, "--speculative-config") == (
        '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
    )
    assert "--enable-chunked-prefill" in plan.vllm_args
    assert "--async-scheduling" in plan.vllm_args
    assert not any(
        name == "VLLM_TUNED_CONFIG_FOLDER" for name, _ in plan.container_env
    )
    assert all(
        mount.container_path != "/vllm-tuned-configs" for mount in plan.mounts
    )


def test_plan_uses_dgx_spark_fp8_arguments(make_plan):
    plan = make_plan("qwen36-fp8", reasoning=True)

    assert _argument_value(plan.vllm_args, "--gpu-memory-utilization") == "0.7"
    assert _argument_value(plan.vllm_args, "--max-num-seqs") == "4"
    assert _argument_value(plan.vllm_args, "--max-num-batched-tokens") == "8192"
    assert _argument_value(plan.vllm_args, "--safetensors-load-strategy") == "lazy"
    assert _argument_value(plan.vllm_args, "--tool-call-parser") == "qwen3_coder"
    assert _argument_value(plan.vllm_args, "--moe-backend") == "triton"
    assert ("VLLM_USE_DEEP_GEMM", "0") in plan.container_env
    assert (
        "VLLM_TUNED_CONFIG_FOLDER",
        "/vllm-tuned-configs",
    ) in plan.container_env
    tuned_mount = next(
        mount
        for mount in plan.mounts
        if mount.container_path == "/vllm-tuned-configs"
    )
    assert tuned_mount.read_only is True
    assert (
        tuned_mount.host_path
        / "E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,"
        "block_shape=[128,128].json"
    ).is_file()
    assert _argument_value(plan.vllm_args, "--speculative-config") == (
        '{"method":"mtp","num_speculative_tokens":2}'
    )


def test_plan_uses_single_spark_mistral4_arguments(make_plan):
    plan = make_plan("mistral4-nvfp4", reasoning=True)

    assert plan.model == "mistralai/Mistral-Small-4-119B-2603-NVFP4"
    assert _argument_value(plan.vllm_args, "--quantization") == (
        "compressed-tensors"
    )
    assert _argument_value(plan.vllm_args, "--gpu-memory-utilization") == "0.8"
    assert _argument_value(plan.vllm_args, "--max-model-len") == "131072"
    assert _argument_value(plan.vllm_args, "--max-num-seqs") == "128"
    assert _argument_value(plan.vllm_args, "--max-num-batched-tokens") == "16384"
    assert _argument_value(plan.vllm_args, "--load-format") == "mistral"
    assert _argument_value(plan.vllm_args, "--tokenizer-mode") == "mistral"
    assert _argument_value(plan.vllm_args, "--config-format") == "mistral"
    assert _argument_value(plan.vllm_args, "--attention-backend") == "TRITON_MLA"
    assert _argument_value(plan.vllm_args, "--reasoning-parser") == "mistral"
    assert _argument_value(plan.vllm_args, "--tool-call-parser") == "mistral"
    assert _argument_value(plan.vllm_args, "--limit-mm-per-prompt") == (
        '{"image":4}'
    )
    assert _argument_value(plan.vllm_args, "--kv-cache-memory-bytes") == (
        "15032385536"
    )
    assert "--skip-mm-profiling" in plan.vllm_args
    assert "--enable-chunked-prefill" in plan.vllm_args
    assert "--moe-backend" not in plan.vllm_args


def test_plan_uses_single_spark_diffusion_gemma_arguments(make_plan):
    plan = make_plan("diffusion-gemma-nvfp4")

    assert plan.model == "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
    assert _argument_value(plan.vllm_args, "--quantization") == "modelopt_fp4"
    assert _argument_value(plan.vllm_args, "--gpu-memory-utilization") == "0.8"
    assert _argument_value(plan.vllm_args, "--max-model-len") == "262144"
    assert _argument_value(plan.vllm_args, "--max-num-seqs") == "4"
    assert _argument_value(plan.vllm_args, "--max-num-batched-tokens") == "8192"
    assert _argument_value(plan.vllm_args, "--load-format") == "fastsafetensors"
    assert _argument_value(plan.vllm_args, "--attention-backend") == "TRITON_ATTN"
    assert _argument_value(plan.vllm_args, "--diffusion-config") == (
        '{"canvas_length":256}'
    )
    assert _argument_value(plan.vllm_args, "--override-generation-config") == (
        '{"max_new_tokens":null}'
    )
    assert _argument_value(plan.vllm_args, "--reasoning-parser") == "gemma4"
    assert _argument_value(plan.vllm_args, "--tool-call-parser") == "gemma4"
    assert _argument_value(plan.vllm_args, "--default-chat-template-kwargs") == (
        '{"enable_thinking":false}'
    )
    assert _argument_value(plan.vllm_args, "--limit-mm-per-prompt") == (
        '{"image":4,"video":1}'
    )
    assert _argument_value(plan.vllm_args, "--mm-processor-kwargs") == (
        '{"max_soft_tokens":280}'
    )
    assert "--enable-auto-tool-choice" in plan.vllm_args
    assert "--enable-chunked-prefill" in plan.vllm_args
    assert "--moe-backend" not in plan.vllm_args
    assert ("VLLM_USE_V2_MODEL_RUNNER", "1") in plan.container_env


def test_diffusion_gemma_reasoning_flag_enables_thinking_by_default(make_plan):
    plan = make_plan("diffusion-gemma-nvfp4", reasoning=True)

    assert _argument_value(plan.vllm_args, "--default-chat-template-kwargs") == (
        '{"enable_thinking":true}'
    )
    assert plan.vllm_args.count("--reasoning-parser") == 1
    assert plan.vllm_args.count("--tool-call-parser") == 1


def test_plan_uses_single_spark_nemotron3_nano_omni_arguments(make_plan):
    plan = make_plan("nemotron3-nano-omni-nvfp4")

    assert plan.model == (
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
    )
    assert _argument_value(plan.vllm_args, "--quantization") == "modelopt_mixed"
    assert _argument_value(plan.vllm_args, "--gpu-memory-utilization") == "0.4"
    assert _argument_value(plan.vllm_args, "--max-model-len") == "131072"
    assert _argument_value(plan.vllm_args, "--max-num-seqs") == "8"
    assert _argument_value(plan.vllm_args, "--max-num-batched-tokens") == "32768"
    assert _argument_value(plan.vllm_args, "--load-format") == "fastsafetensors"
    assert _argument_value(plan.vllm_args, "--kv-cache-dtype") == "fp8"
    assert _argument_value(plan.vllm_args, "--video-pruning-rate") == "0.5"
    assert _argument_value(plan.vllm_args, "--reasoning-parser") == "nemotron_v3"
    assert _argument_value(plan.vllm_args, "--tool-call-parser") == "qwen3_coder"
    assert _argument_value(plan.vllm_args, "--default-chat-template-kwargs") == (
        '{"enable_thinking":false}'
    )
    assert _argument_value(plan.vllm_args, "--limit-mm-per-prompt") == (
        '{"video":1,"image":1,"audio":1}'
    )
    assert _argument_value(plan.vllm_args, "--media-io-kwargs") == (
        '{"video":{"fps":2,"num_frames":256}}'
    )
    assert "--enable-auto-tool-choice" in plan.vllm_args
    assert "--enable-chunked-prefill" in plan.vllm_args
    assert "--allowed-local-media-path" not in plan.vllm_args
    assert "--moe-backend" not in plan.vllm_args
    assert plan.startup_python_packages == (
        "av==18.0.0",
        "scipy==1.18.0",
        "soundfile==0.14.0",
        "soxr==1.1.0",
    )
    assert any(
        mount.container_path == "/root/.cache/vllm" and not mount.read_only
        for mount in plan.mounts
    )
    assert all(mount.container_path != "/root/.cache/pip" for mount in plan.mounts)


def test_nemotron3_nano_omni_reasoning_flag_enables_thinking(make_plan):
    plan = make_plan("nemotron3-nano-omni-nvfp4", reasoning=True)

    assert _argument_value(plan.vllm_args, "--default-chat-template-kwargs") == (
        '{"enable_thinking":true}'
    )
    assert plan.vllm_args.count("--reasoning-parser") == 1
    assert plan.vllm_args.count("--tool-call-parser") == 1


def test_diffusion_parser_defaults_do_not_change_other_profiles(make_plan):
    for variant in (
        "qwen36-fp8",
        "qwen36-nvfp4",
        "gemma4-nvfp4",
        "ornith-nvfp4",
        "mistral4-nvfp4",
    ):
        args = make_plan(variant).vllm_args
        assert "--reasoning-parser" not in args
        assert "--tool-call-parser" not in args
        assert "--default-chat-template-kwargs" not in args


def test_plan_uses_compressed_tensors_for_ornith(make_plan):
    plan = make_plan("ornith-nvfp4", reasoning=True)

    assert _argument_value(plan.vllm_args, "--quantization") == "compressed-tensors"
    assert _argument_value(plan.vllm_args, "--gpu-memory-utilization") == "0.7"
    assert _argument_value(plan.vllm_args, "--max-num-seqs") == "4"
    assert _argument_value(plan.vllm_args, "--max-num-batched-tokens") == "8192"
    assert _argument_value(plan.vllm_args, "--tool-call-parser") == "qwen3_xml"
    assert "--moe-backend" not in plan.vllm_args


def test_plan_uses_variant_specific_gemma_arguments(make_plan):
    plan = make_plan("gemma4-nvfp4", reasoning=True)

    assert "--moe-backend" not in plan.vllm_args
    assert _argument_value(plan.vllm_args, "--quantization") == "modelopt_fp4"
    assert _argument_value(plan.vllm_args, "--gpu-memory-utilization") == "0.8"
    assert _argument_value(plan.vllm_args, "--max-num-seqs") == "32"
    assert _argument_value(plan.vllm_args, "--max-num-batched-tokens") == "16384"
    assert _argument_value(plan.vllm_args, "--kv-cache-dtype") == "bfloat16"
    assert _argument_value(plan.vllm_args, "--reasoning-parser") == "gemma4"
    assert _argument_value(plan.vllm_args, "--tool-call-parser") == "gemma4"
    assert _argument_value(plan.vllm_args, "--chat-template").endswith(
        "tool_chat_template_gemma4.jinja"
    )
    assert _argument_value(plan.vllm_args, "--limit-mm-per-prompt") == (
        '{"image":4,"video":0}'
    )
    assert "--async-scheduling" in plan.vllm_args


def test_plan_uses_latest_profile_backend_defaults(make_plan):
    assert (
        _argument_value(
            make_plan("qwen36-nvfp4").vllm_args,
            "--moe-backend",
        )
        == "marlin"
    )
    assert "--moe-backend" not in make_plan("ornith-nvfp4").vllm_args
    assert "--moe-backend" not in make_plan("gemma4-nvfp4").vllm_args


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_ready_timeout_must_be_a_positive_integer(make_plan, value: str):
    with pytest.raises(ConfigurationError, match="VLLM_READY_TIMEOUT"):
        make_plan(env_overrides={"VLLM_READY_TIMEOUT": value})


@pytest.mark.parametrize("value", ["0", "65536", "bad"])
def test_host_port_is_validated(make_plan, value: str):
    with pytest.raises(ConfigurationError, match="VLLM_HOST_PORT"):
        make_plan(env_overrides={"VLLM_HOST_PORT": value})


def test_host_port_drives_endpoint_and_docker_plan(make_plan):
    plan = make_plan(env_overrides={"VLLM_HOST_PORT": "9000"})

    assert plan.host_port == 9000
    assert plan.base_url == "http://127.0.0.1:9000"
    assert _argument_value(plan.vllm_args, "--port") == "8000"


def test_warmup_count_must_not_be_negative(make_plan):
    with pytest.raises(ConfigurationError, match="VLLM_WARMUP_REQUESTS"):
        make_plan(env_overrides={"VLLM_WARMUP_REQUESTS": "-1"})


def test_no_warmup_does_not_parse_unused_warmup_environment(make_plan):
    plan = make_plan(
        no_warmup=True,
        env_overrides={"VLLM_WARMUP_REQUESTS": "invalid"},
    )

    assert plan.warmup_requests == 0


@pytest.mark.parametrize(
    "policy",
    ["always", "unless-stopped", "on-failure", "on-failure:3", "no"],
)
def test_valid_restart_policies_are_resolved(make_plan, policy: str):
    assert make_plan(restart_policy=policy).restart_policy == policy


def test_invalid_restart_policy_is_rejected(make_plan):
    with pytest.raises(ConfigurationError, match="restart policy"):
        make_plan(restart_policy="explode-forever")


def test_environment_settings_are_collected_in_plan(make_plan, tmp_path: Path):
    cache = tmp_path / "custom-cache"
    hf_cache = tmp_path / "custom-hf-cache"
    artifacts = tmp_path / "custom-artifacts"
    plan = make_plan(
        "qwen36-fp8",
        env_overrides={
            "VLLM_CACHE_DIR": str(cache),
            "VLLM_HF_CACHE_DIR": str(hf_cache),
            "VLLM_ARTIFACT_DIR": str(artifacts),
            "VLLM_SAFETENSORS_LOAD_STRATEGY": "prefetch",
            "VLLM_MARLIN_USE_ATOMIC_ADD": "0",
        },
    )

    assert artifacts == plan.artifact_dir
    assert {
        mount.host_path for mount in plan.mounts if not mount.read_only
    } == {cache, hf_cache}
    assert _argument_value(plan.vllm_args, "--safetensors-load-strategy") == "prefetch"
    assert ("VLLM_MARLIN_USE_ATOMIC_ADD", "0") in plan.container_env


def test_variant_is_required_to_resolve_launch_plan():
    with pytest.raises(ConfigurationError, match="variant is required"):
        resolve_launch_plan(LaunchArgs(variant=None, show_defaults=True), {})


def _argument_value(arguments: tuple[str, ...], name: str) -> str:
    index = arguments.index(name)
    return arguments[index + 1]
