from __future__ import annotations

import os
import signal
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

from .cli import LaunchArgs, parse_args
from .config import VARIANTS, resolve_variant_profile
from .docker_ops import DockerRuntime
from .http_ops import VllmClient
from .launcher import Clock, ContainerRuntime, Launcher, SecretProvider, VllmService
from .plan import resolve_launch_plan
from .presentation import Reporter, RichReporter
from .secrets import HuggingFaceTokenProvider

ClientFactory = Callable[[str], VllmService]


def run(
    args: LaunchArgs,
    *,
    env: Mapping[str, str] | None = None,
    runtime: ContainerRuntime | None = None,
    reporter: Reporter | None = None,
    secret_provider: SecretProvider | None = None,
    client_factory: ClientFactory | None = None,
    clock: Clock | None = None,
) -> int:
    """Compose dependencies and run one operation without changing signal handlers."""

    resolved_env = os.environ if env is None else env
    resolved_reporter = RichReporter() if reporter is None else reporter
    try:
        if args.show_defaults:
            resolved_reporter.show_defaults(
                resolve_variant_profile(variant) for variant in VARIANTS
            )
            return 0

        plan = resolve_launch_plan(args, resolved_env)
        resolved_runtime = DockerRuntime() if runtime is None else runtime
        resolved_secrets = (
            HuggingFaceTokenProvider(resolved_env)
            if secret_provider is None
            else secret_provider
        )
        resolved_client_factory = (
            (lambda base_url: VllmClient(base_url))
            if client_factory is None
            else client_factory
        )
        launcher = Launcher(
            runtime=resolved_runtime,
            client=resolved_client_factory(plan.base_url),
            reporter=resolved_reporter,
            secret_provider=resolved_secrets,
            clock=clock,
        )
        return launcher.launch(plan)
    except KeyboardInterrupt:
        resolved_reporter.warning("Interrupted. Exiting...")
        return 130
    except Exception as exc:
        resolved_reporter.error(f"Error: {exc}")
        return 1


def _signal_to_keyboard_interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


@contextmanager
def cli_signal_handlers() -> Iterator[None]:
    """Install CLI-only handlers and restore the process state afterward."""

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_to_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _signal_to_keyboard_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv: list[str] | None = None) -> int:
    with cli_signal_handlers():
        return run(parse_args(argv))
