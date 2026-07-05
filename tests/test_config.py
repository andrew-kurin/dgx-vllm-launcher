from __future__ import annotations

from dgx_vllm_launcher.config import (
    DEFAULT_READY_TIMEOUT,
    DEFAULT_FP8_IMAGE,
    DEFAULT_NVFP4_IMAGE,
    resolve_variant_config,
    resolve_cache_dir,
)


def test_resolve_variant_config_fp8_defaults():
    cfg = resolve_variant_config("fp8", env_getter=lambda key, default: default)
    assert cfg.model == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert cfg.image == DEFAULT_FP8_IMAGE
    assert cfg.served_model_name == "qwen36-fp8"
    assert cfg.ready_timeout_seconds == DEFAULT_READY_TIMEOUT


def test_resolve_variant_config_nvfp4_defaults_and_unified_env_override():
    cfg = resolve_variant_config(
        "nvfp4",
        env_getter=lambda key, default: {
            "VLLM_READY_TIMEOUT": "42",
        }.get(key, default),
    )
    assert cfg.model == "/model"
    assert cfg.image == DEFAULT_NVFP4_IMAGE
    assert cfg.served_model_name == "qwen36-nvfp4"
    assert cfg.ready_timeout_seconds == 42


def test_resolve_cache_dir_uses_env_override():
    value = resolve_cache_dir(env_getter=lambda key, default: "/tmp/custom-cache")
    assert value == "/tmp/custom-cache"


def test_resolve_variant_config_invalid_timeout_raises():
    try:
        resolve_variant_config(
            "fp8",
            env_getter=lambda key, default: "not-an-int" if key == "VLLM_READY_TIMEOUT" else default,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid timeout should raise")


def test_variant_specific_timeout_envs_are_supported_for_backward_compat():
    cfg_fp8 = resolve_variant_config(
        "fp8",
        env_getter=lambda key, default: {
            "VLLM_READY_TIMEOUT_FP8": "11",
        }.get(key, default),
    )
    cfg_nvfp4 = resolve_variant_config(
        "nvfp4",
        env_getter=lambda key, default: {
            "VLLM_READY_TIMEOUT_NVFP4": "22",
        }.get(key, default),
    )

    assert cfg_fp8.ready_timeout_seconds == 11
    assert cfg_nvfp4.ready_timeout_seconds == 22
