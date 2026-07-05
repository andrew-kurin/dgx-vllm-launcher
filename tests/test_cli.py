from __future__ import annotations

from dgx_vllm_launcher import cli


def test_parse_args_qwen36_fp8_defaults():
    args = cli.parse_args(["qwen36-fp8"])

    assert args.variant == "qwen36-fp8"
    assert args.reasoning is False
    assert args.no_warmup is False
    assert args.no_smoke_check is False
    assert args.enable_prefix_caching is False
    assert args.detach is False
    assert args.moe_backend is None
    assert args.linear_backend is None
    assert args.restart_policy is None
    assert args.show_defaults is False


def test_parse_args_flags():
    args = cli.parse_args(
        [
            "qwen36-nvfp4",
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

    assert args.variant == "qwen36-nvfp4"
    assert args.reasoning is True
    assert args.no_warmup is True
    assert args.no_smoke_check is True
    assert args.detach is True
    assert args.enable_prefix_caching is True
    assert args.moe_backend == "flashinfer_b12x"
    assert args.linear_backend == "flashinfer"
    assert args.restart_policy == "on-failure"
    assert args.show_defaults is False


def test_parse_args_gemma4_variant():
    args = cli.parse_args(["gemma4-nvfp4", "--restart-policy", "unless-stopped"])
    assert args.variant == "gemma4-nvfp4"
    assert args.restart_policy == "unless-stopped"
    assert args.show_defaults is False


def test_parse_args_ornith_variant():
    args = cli.parse_args(["ornith-nvfp4", "--moe-backend", "flashinfer"])
    assert args.variant == "ornith-nvfp4"
    assert args.moe_backend == "flashinfer"
    assert args.show_defaults is False


def test_parse_args_short_flags():
    args = cli.parse_args(
        [
            "qwen36-nvfp4",
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

    assert args.variant == "qwen36-nvfp4"
    assert args.reasoning is True
    assert args.no_warmup is True
    assert args.no_smoke_check is True
    assert args.detach is True
    assert args.enable_prefix_caching is True
    assert args.moe_backend == "flashinfer_b12x"
    assert args.linear_backend == "flashinfer"
    assert args.restart_policy == "unless-stopped"
    assert args.show_defaults is False


def test_parse_args_show_defaults_flag_without_variant():
    args = cli.parse_args(["--show-defaults"])
    assert args.variant is None
    assert args.show_defaults is True


def test_parse_args_show_defaults_with_variant():
    args = cli.parse_args(["qwen36-fp8", "--show-defaults"])
    assert args.show_defaults is True
    assert args.variant == "qwen36-fp8"


def test_parse_args_rejects_missing_variant_for_launch():
    try:
        cli.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parse_args to fail")


def test_parse_args_rejects_invalid_variant():
    try:
        cli.parse_args(["bad"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected parse_args to fail")
