from __future__ import annotations

from collections.abc import Callable, Iterable
from threading import Event, Lock, Thread, current_thread
from typing import Protocol

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import VariantProfile
from .plan import LaunchPlan


class Reporter(Protocol):
    def show_plan(self, plan: LaunchPlan) -> None: ...

    def show_defaults(self, profiles: Iterable[VariantProfile]) -> None: ...

    def info(self, message: str) -> None: ...

    def success(self, message: str) -> None: ...

    def warning(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...

    def container_log(self, line: str) -> None: ...

    def startup_logs(self, logs: str) -> None: ...


class LogStream(Protocol):
    def lines(self) -> Iterable[str]: ...

    def close(self) -> None: ...


class LogTailer(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class BackgroundLogTailer:
    def __init__(
        self,
        stream_factory: Callable[[], LogStream],
        emit: Callable[[str], None],
    ) -> None:
        self._stream_factory = stream_factory
        self._emit = emit
        self._lock = Lock()
        self._stream: LogStream | None = None
        self._stop_event: Event | None = None
        self._thread: Thread | None = None
        self._stopping = False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None or self._stopping:
                return

            stream = self._stream_factory()
            stop_event = Event()

            def read_lines() -> None:
                for line in stream.lines():
                    if stop_event.is_set():
                        return
                    if line:
                        self._emit(line)

            thread = Thread(target=read_lines, daemon=True)
            self._stream = stream
            self._stop_event = stop_event
            self._thread = thread
            try:
                thread.start()
            except Exception:
                self._stream = None
                self._stop_event = None
                self._thread = None
                stop_event.set()
                stream.close()
                raise

    def stop(self) -> None:
        with self._lock:
            if self._thread is None or self._stopping:
                return
            stream = self._stream
            stop_event = self._stop_event
            thread = self._thread
            self._stopping = True

        assert stream is not None
        assert stop_event is not None
        stop_event.set()
        try:
            stream.close()
        finally:
            try:
                if thread.is_alive() and thread is not current_thread():
                    thread.join(timeout=1)
            finally:
                with self._lock:
                    if self._thread is thread:
                        self._stream = None
                        self._stop_event = None
                        self._thread = None
                    self._stopping = False


class RichReporter:
    def __init__(self, console: Console | None = None) -> None:
        self.console = Console() if console is None else console

    def show_plan(self, plan: LaunchPlan) -> None:
        settings = Table(title="Launcher settings", box=box.ROUNDED)
        settings.add_column("Setting", style="cyan")
        settings.add_column("Value", style="magenta")
        settings.add_row("Variant", plan.variant)
        settings.add_row("Model", plan.model)
        if plan.model != plan.configured_model:
            settings.add_row("Configured HF model", plan.configured_model)
        settings.add_row("Image", plan.image)
        settings.add_row("Container", plan.container_name)
        settings.add_row(
            "Docker bind",
            f"{plan.docker_bind_address}:{plan.host_port}",
        )
        settings.add_row("Endpoint", plan.base_url)
        settings.add_row("Served model name", plan.served_model_name)
        settings.add_row(
            "Preloaded model", "yes" if plan.uses_preloaded_model else "no"
        )
        if plan.preloaded_model_path is not None:
            settings.add_row("Preloaded path", str(plan.preloaded_model_path))
        settings.add_row("Ready timeout", f"{plan.ready_timeout_seconds}s")
        settings.add_row("Warmup requests", str(plan.warmup_requests))
        settings.add_row(
            "Smoke check", "enabled" if plan.run_smoke_check else "disabled"
        )
        settings.add_row("Restart policy", plan.restart_policy or "(none)")
        if plan.startup_python_packages:
            settings.add_row(
                "Startup Python packages",
                ", ".join(plan.startup_python_packages),
            )
        self.console.print(Panel(settings, title="vLLM serve", expand=False))

        arguments = Table(
            title="vLLM arguments",
            box=box.ROUNDED,
            show_header=False,
            show_edge=False,
            padding=(0, 1),
        )
        arguments.add_column("Argument", style="cyan")
        arguments.add_column("Value", style="magenta")
        for argument, value in _argument_rows(plan.vllm_args):
            arguments.add_row(argument, value)
        self.console.print(arguments)

    def show_defaults(self, profiles: Iterable[VariantProfile]) -> None:
        table = Table(title="Default launch configuration", box=box.ROUNDED)
        table.add_column("Variant", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("MoE backend", style="yellow")
        table.add_column("Linear backend", style="yellow")
        table.add_column("Quantization", style="blue")
        for profile in profiles:
            table.add_row(
                profile.variant,
                profile.model,
                profile.default_moe_backend or "(none)",
                profile.default_linear_backend or "(none)",
                profile.quantization or "(none)",
            )
        self.console.print(Panel(table, title="Recommended defaults", expand=False))

    def info(self, message: str) -> None:
        self.console.print(Text(message, style="blue"))

    def success(self, message: str) -> None:
        self.console.print(Text(message, style="green"))

    def warning(self, message: str) -> None:
        self.console.print(Text(message, style="yellow"))

    def error(self, message: str) -> None:
        self.console.print(Text(message, style="red"))

    def container_log(self, line: str) -> None:
        self.console.print(Text(line, style="dim"))

    def startup_logs(self, logs: str) -> None:
        self.console.print(
            Panel(Text(logs), title="startup logs", border_style="red", expand=False)
        )


def _argument_rows(arguments: tuple[str, ...]) -> Iterable[tuple[str, str]]:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
            yield argument, arguments[index + 1]
            index += 2
        else:
            yield argument, ""
            index += 1
