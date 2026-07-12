from __future__ import annotations

from io import StringIO

from rich.console import Console

from dgx_vllm_launcher.config import VARIANT_PROFILES
from dgx_vllm_launcher.presentation import RichReporter


def test_summary_renders_the_fully_resolved_plan(make_plan):
    output = StringIO()
    reporter = RichReporter(
        Console(file=output, color_system=None, force_terminal=False, width=140)
    )
    plan = make_plan(
        moe_backend="custom-moe",
        linear_backend="custom-linear",
        env_overrides={"VLLM_HOST_PORT": "9000"},
    )

    reporter.show_plan(plan)

    rendered = output.getvalue()
    assert "http://127.0.0.1:9000" in rendered
    assert "--moe-backend" in rendered and "custom-moe" in rendered
    assert "--linear-backend" in rendered and "custom-linear" in rendered


def test_default_profiles_include_latest_variant_settings():
    output = StringIO()
    reporter = RichReporter(
        Console(file=output, color_system=None, force_terminal=False, width=160)
    )

    reporter.show_defaults(VARIANT_PROFILES.values())

    rendered = output.getvalue()
    assert "nvidia/Qwen3.6-35B-A3B-NVFP4" in rendered
    assert "marlin" in rendered
    assert "nvidia/Gemma-4-26B-A4B-NVFP4" in rendered
    assert "nvidia/diffusiongemma-26B-A4B-it-NVFP4" in rendered


def test_container_logs_are_rendered_as_text_not_rich_markup():
    output = StringIO()
    reporter = RichReporter(
        Console(file=output, color_system=None, force_terminal=False)
    )

    reporter.container_log("[red]literal container output[/red]")

    assert output.getvalue().strip() == "[red]literal container output[/red]"
