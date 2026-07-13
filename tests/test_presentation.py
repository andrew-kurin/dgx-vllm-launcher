from __future__ import annotations

from collections.abc import Iterable
from io import StringIO
from threading import Event, Lock, Thread

from rich.console import Console

from dgx_vllm_launcher.config import VARIANT_PROFILES
from dgx_vllm_launcher.presentation import BackgroundLogTailer, RichReporter


class BlockingLogStream:
    def __init__(self) -> None:
        self.waiting = Event()
        self.release = Event()
        self.closed = Event()
        self.close_calls = 0

    def lines(self) -> Iterable[str]:
        yield "before stop"
        self.waiting.set()
        self.release.wait(timeout=2)
        yield "after stop"

    def close(self) -> None:
        self.close_calls += 1
        self.closed.set()
        self.release.set()


class EmptyLogStream:
    def lines(self) -> Iterable[str]:
        return ()

    def close(self) -> None:
        pass


def test_summary_renders_the_fully_resolved_plan(make_plan):
    output = StringIO()
    reporter = RichReporter(
        Console(file=output, color_system=None, force_terminal=False, width=140)
    )
    plan = make_plan(
        moe_backend="custom-moe",
        linear_backend="custom-linear",
        env_overrides={
            "VLLM_BIND_ADDRESS": "0.0.0.0",
            "VLLM_HOST_PORT": "9000",
        },
    )

    reporter.show_plan(plan)

    rendered = output.getvalue()
    assert "0.0.0.0:9000" in rendered
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
    assert (
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4" in rendered
    )


def test_container_logs_are_rendered_as_text_not_rich_markup():
    output = StringIO()
    reporter = RichReporter(
        Console(file=output, color_system=None, force_terminal=False)
    )

    reporter.container_log("[red]literal container output[/red]")

    assert output.getvalue().strip() == "[red]literal container output[/red]"


def test_background_log_tailer_stop_closes_and_joins_blocked_reader():
    stream = BlockingLogStream()
    emitted: list[str] = []
    tailer = BackgroundLogTailer(lambda: stream, emitted.append)

    tailer.start()
    assert stream.waiting.wait(timeout=1)

    tailer.stop()
    tailer.stop()

    assert stream.closed.is_set()
    assert stream.close_calls == 1
    assert emitted == ["before stop"]


def test_background_log_tailer_serializes_concurrent_starts():
    factory_entered = Event()
    second_factory_entered = Event()
    release_factory = Event()
    calls_lock = Lock()
    factory_calls = 0

    def stream_factory() -> EmptyLogStream:
        nonlocal factory_calls
        with calls_lock:
            factory_calls += 1
            if factory_calls == 1:
                factory_entered.set()
            else:
                second_factory_entered.set()
        release_factory.wait(timeout=2)
        return EmptyLogStream()

    tailer = BackgroundLogTailer(stream_factory, lambda _line: None)
    first_start = Thread(target=tailer.start)
    second_start = Thread(target=tailer.start)

    first_start.start()
    assert factory_entered.wait(timeout=1)
    second_start.start()
    second_factory_entered.wait(timeout=0.25)
    release_factory.set()
    first_start.join(timeout=1)
    second_start.join(timeout=1)

    assert not first_start.is_alive()
    assert not second_start.is_alive()
    assert factory_calls == 1
    tailer.stop()
