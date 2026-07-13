from __future__ import annotations

import pytest

from dgx_vllm_launcher import cli
from dgx_vllm_launcher.plan import LaunchArgs


def test_parse_args_defaults():
    args = cli.parse_args(["qwen36-fp8"])

    assert isinstance(args, LaunchArgs)
    assert args.variant == "qwen36-fp8"
    assert args.reasoning is False
    assert args.no_warmup is False
    assert args.no_smoke_check is False
    assert args.detach is False
    assert args.moe_backend is None
    assert args.linear_backend is None
    assert args.restart_policy is None
    assert args.use_preloaded_models is False
    assert args.preloaded_models_dir is None
    assert args.show_defaults is False


def test_parse_args_long_flags():
    args = cli.parse_args(
        [
            "qwen36-nvfp4",
            "--reasoning",
            "--no-warmup",
            "--no-smoke-check",
            "--detach",
            "--moe-backend",
            "triton",
            "--linear-backend",
            "flashinfer",
            "--restart-policy",
            "on-failure:3",
            "--use-preloaded-models",
            "--preloaded-models-dir",
            "/opt/models",
        ]
    )

    assert args.reasoning is True
    assert args.no_warmup is True
    assert args.no_smoke_check is True
    assert args.detach is True
    assert args.moe_backend == "triton"
    assert args.linear_backend == "flashinfer"
    assert args.restart_policy == "on-failure:3"
    assert args.use_preloaded_models is True
    assert args.preloaded_models_dir == "/opt/models"


def test_parse_args_short_flags():
    args = cli.parse_args(
        [
            "ornith-nvfp4",
            "-r",
            "-w",
            "-s",
            "-d",
            "-m",
            "triton",
            "-l",
            "flashinfer",
            "-R",
            "unless-stopped",
        ]
    )

    assert args.variant == "ornith-nvfp4"
    assert args.reasoning is True
    assert args.no_warmup is True
    assert args.no_smoke_check is True
    assert args.detach is True


def test_parse_args_accepts_mistral4_nvfp4_variant():
    args = cli.parse_args(["mistral4-nvfp4", "--reasoning"])

    assert args.variant == "mistral4-nvfp4"
    assert args.reasoning is True


def test_parse_args_accepts_diffusion_gemma_nvfp4_variant():
    args = cli.parse_args(["diffusion-gemma-nvfp4", "--reasoning"])

    assert args.variant == "diffusion-gemma-nvfp4"
    assert args.reasoning is True


def test_parse_args_accepts_nemotron3_nano_omni_nvfp4_variant():
    args = cli.parse_args(["nemotron3-nano-omni-nvfp4", "--reasoning"])

    assert args.variant == "nemotron3-nano-omni-nvfp4"
    assert args.reasoning is True


def test_parse_args_show_defaults_without_variant():
    args = cli.parse_args(["--show-defaults"])

    assert args.variant is None
    assert args.show_defaults is True


def test_parse_args_show_defaults_with_variant():
    args = cli.parse_args(["gemma4-nvfp4", "--show-defaults"])

    assert args.variant == "gemma4-nvfp4"
    assert args.show_defaults is True


def test_parse_args_rejects_missing_variant_for_launch():
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args([])

    assert exc_info.value.code == 2


def test_parse_args_rejects_invalid_variant():
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args(["bad"])

    assert exc_info.value.code == 2


def test_removed_prefix_caching_compatibility_flag_is_rejected():
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args(["qwen36-fp8", "--enable-prefix-caching"])

    assert exc_info.value.code == 2
