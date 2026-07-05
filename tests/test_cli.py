from __future__ import annotations

from qwen_vllm_launcher import cli


def test_parse_args_fp8_defaults():
    args = cli.parse_args(["fp8"])

    assert args.variant == "fp8"
    assert args.reasoning is False
    assert args.no_warmup is False
    assert args.no_smoke_check is False
    assert args.enable_prefix_caching is False
    assert args.detach is False
    assert args.moe_backend is None
    assert args.linear_backend is None
    assert args.restart_policy is None


def test_parse_args_flags():
    args = cli.parse_args(
        [
            "nvfp4",
            "--reasoning",
            "--no-warmup",
            "--no-smoke-check",
            "--detach",
            "--enable-prefix-caching",
            "--moe-backend",
            "flashinfer_b12x",
            "--linear-backend",
            "flashinfer",
            "--restart-policy",
            "on-failure",
        ]
    )

    assert args.variant == "nvfp4"
    assert args.reasoning is True
    assert args.no_warmup is True
    assert args.no_smoke_check is True
    assert args.detach is True
    assert args.enable_prefix_caching is True
    assert args.moe_backend == "flashinfer_b12x"
    assert args.linear_backend == "flashinfer"
    assert args.restart_policy == "on-failure"


def test_parse_args_short_flags():
    args = cli.parse_args(
        [
            "nvfp4",
            "-r",
            "-w",
            "-s",
            "-d",
            "-p",
            "-m",
            "flashinfer_b12x",
            "-l",
            "flashinfer",
            "-R",
            "unless-stopped",
        ]
    )

    assert args.variant == "nvfp4"
    assert args.reasoning is True
    assert args.no_warmup is True
    assert args.no_smoke_check is True
    assert args.detach is True
    assert args.enable_prefix_caching is True
    assert args.moe_backend == "flashinfer_b12x"
    assert args.linear_backend == "flashinfer"
    assert args.restart_policy == "unless-stopped"


def test_parse_args_rejects_invalid_variant():
    try:
        cli.parse_args(["bad"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parse_args to fail")
