from __future__ import annotations

import os

from dgx_vllm_launcher.config import (
    DEFAULT_READY_TIMEOUT,
    DEFAULT_FP8_IMAGE,
    DEFAULT_NVFP4_IMAGE,
    DEFAULT_PRELOADED_MODELS_DIR,
    DEFAULT_GEMMA4_NVFP4_IMAGE,
    DEFAULT_ORNITH_NVFP4_IMAGE,
    VARIANTS,
    VARIANT_PROFILES,
    resolve_variant_config,
    resolve_cache_dir,
    resolve_preloaded_models_root,
)


def test_resolve_variant_config_qwen36_fp8_defaults():
    cfg = resolve_variant_config("qwen36-fp8", env_getter=lambda key, default: default)
    assert cfg.model == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert cfg.image == DEFAULT_FP8_IMAGE
    assert cfg.served_model_name == "qwen36-fp8"
    assert cfg.ready_timeout_seconds == DEFAULT_READY_TIMEOUT


def test_resolve_variant_config_qwen36_nvfp4_defaults_and_unified_env_override():
    cfg = resolve_variant_config(
        "qwen36-nvfp4",
        env_getter=lambda key, default: {
            "VLLM_READY_TIMEOUT": "42",
        }.get(key, default),
    )
    assert cfg.model == "Qwen/Qwen3.6-35B-A3B-NVFP4"
    assert cfg.image == DEFAULT_NVFP4_IMAGE
    assert cfg.served_model_name == "qwen36-nvfp4"
    assert cfg.ready_timeout_seconds == 42


def test_resolve_variant_config_gemma4_defaults():
    cfg = resolve_variant_config("gemma4-nvfp4", env_getter=lambda key, default: default)
    assert cfg.model == "nvidia/Gemma-4-26B-A4B-NVFP4"
    assert cfg.image == DEFAULT_GEMMA4_NVFP4_IMAGE
    assert cfg.served_model_name == "gemma4-nvfp4"
    assert cfg.ready_timeout_seconds == DEFAULT_READY_TIMEOUT


def test_resolve_variant_config_ornith_defaults():
    cfg = resolve_variant_config("ornith-nvfp4", env_getter=lambda key, default: default)
    assert cfg.model == "sakamakismile/Ornith-1.0-35B-NVFP4"
    assert cfg.image == DEFAULT_ORNITH_NVFP4_IMAGE
    assert cfg.served_model_name == "ornith-nvfp4"
    assert cfg.ready_timeout_seconds == DEFAULT_READY_TIMEOUT


def test_resolve_variant_config_qwen36_fp8_respects_unified_timeout():
    cfg = resolve_variant_config(
        "qwen36-fp8",
        env_getter=lambda key, default: {
            "VLLM_READY_TIMEOUT": "64",
        }.get(key, default),
    )

    assert cfg.ready_timeout_seconds == 64


def test_variant_profiles_are_complete():
    assert set(VARIANTS) == set(VARIANT_PROFILES)


def test_variant_profiles_capture_expected_launch_hints():
    assert VARIANT_PROFILES["qwen36-fp8"].requires_hf_token is True
    assert VARIANT_PROFILES["qwen36-fp8"].quantization is None
    assert VARIANT_PROFILES["qwen36-fp8"].inject_hf_token is True

    assert VARIANT_PROFILES["qwen36-nvfp4"].quantization == "modelopt"
    assert VARIANT_PROFILES["qwen36-nvfp4"].mount_local_model is True
    assert VARIANT_PROFILES["qwen36-nvfp4"].default_moe_backend == "flashinfer_b12x"
    assert VARIANT_PROFILES["qwen36-nvfp4"].inject_hf_token is False
    assert VARIANT_PROFILES["qwen36-nvfp4"].model.startswith("Qwen/Qwen3.6")

    assert VARIANT_PROFILES["gemma4-nvfp4"].requires_hf_token is False
    assert VARIANT_PROFILES["gemma4-nvfp4"].default_moe_backend is None
    assert VARIANT_PROFILES["gemma4-nvfp4"].runtime_defaults.reasoning_parser == "gemma4"
    assert VARIANT_PROFILES["gemma4-nvfp4"].runtime_defaults.tool_call_parser == "gemma4"
    assert VARIANT_PROFILES["gemma4-nvfp4"].inject_hf_token is True
    assert VARIANT_PROFILES["gemma4-nvfp4"].mount_local_model is True

    assert VARIANT_PROFILES["ornith-nvfp4"].requires_hf_token is False
    assert VARIANT_PROFILES["ornith-nvfp4"].default_moe_backend is None
    assert VARIANT_PROFILES["ornith-nvfp4"].inject_hf_token is True
    assert VARIANT_PROFILES["ornith-nvfp4"].mount_local_model is True
    assert VARIANT_PROFILES["ornith-nvfp4"].quantization == "modelopt"

def test_resolve_preloaded_models_root_default():
    value = resolve_preloaded_models_root(env_getter=lambda key, default: default)
    assert value == os.path.expanduser(DEFAULT_PRELOADED_MODELS_DIR)


def test_resolve_preloaded_models_root_override():
    value = resolve_preloaded_models_root(
        override_root="/tmp/models",
        env_getter=lambda key, default: default,
    )
    assert value == "/tmp/models"


def test_resolve_cache_dir_uses_env_override():
    value = resolve_cache_dir(env_getter=lambda key, default: "/tmp/custom-cache")
    assert value == "/tmp/custom-cache"


def test_resolve_variant_config_invalid_timeout_raises():
    try:
        resolve_variant_config(
            "qwen36-fp8",
            env_getter=lambda key, default: "not-an-int" if key == "VLLM_READY_TIMEOUT" else default,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid timeout should raise")
