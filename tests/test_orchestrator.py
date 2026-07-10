from __future__ import annotations

import signal
from collections.abc import Iterable, Mapping

import pytest

import dgx_vllm_launcher.orchestrator as orchestrator
from dgx_vllm_launcher.cli import LaunchArgs
from dgx_vllm_launcher.config import VariantProfile
from dgx_vllm_launcher.http_ops import HttpResult
from dgx_vllm_launcher.launcher import (
    LaunchError,
    Launcher,
    ReadinessError,
    run_warmup,
    smoke_check,
    wait_for_health,
)
from dgx_vllm_launcher.plan import LaunchPlan


class FakeLogStream:
    def __init__(self, *, lines: Iterable[str] = (), returncode: int = 0) -> None:
        self._lines = tuple(lines)
        self._returncode = returncode
        self.closed = False

    def lines(self) -> Iterable[str]:
        return iter(self._lines)

    def wait(self) -> int:
        return self._returncode

    def close(self) -> None:
        self.closed = True


class FakeRuntime:
    def __init__(
        self,
        *,
        exists: bool = False,
        managed: bool = True,
        running: bool = True,
        foreground_returncode: int = 0,
        prepare_error: Exception | None = None,
        start_error_after_create: Exception | None = None,
    ) -> None:
        self.exists = exists
        self.managed = managed
        self.running = running
        self.foreground_returncode = foreground_returncode
        self.prepare_error = prepare_error
        self.start_error_after_create = start_error_after_create
        self.started_hf_token: str | None = None
        self.events: list[str] = []
        self.log_streams: list[FakeLogStream] = []

    def prepare(self, plan: LaunchPlan) -> None:
        self.events.append("prepare")
        if self.prepare_error:
            raise self.prepare_error

    def container_exists(self, name: str) -> bool:
        self.events.append("exists")
        return self.exists

    def container_is_managed(self, name: str) -> bool:
        self.events.append("managed")
        return self.managed

    def container_running(self, name: str, *, timeout: float) -> bool:
        self.events.append(f"running:{timeout:g}")
        return self.running

    def start(self, plan: LaunchPlan, *, hf_token: str | None) -> str:
        self.events.append("start")
        self.started_hf_token = hf_token
        self.exists = True
        self.running = True
        if self.start_error_after_create:
            raise self.start_error_after_create
        return "container-id"

    def logs(self, name: str, *, tail: int = 200) -> str:
        self.events.append(f"logs:{tail}")
        return "startup logs"

    def stop(self, name: str) -> None:
        self.events.append("stop")
        self.running = False

    def remove(self, name: str) -> None:
        self.events.append("remove")
        self.exists = False

    def open_logs(
        self,
        name: str,
        *,
        capture_output: bool,
        tail: int | None = None,
    ) -> FakeLogStream:
        self.events.append(f"open-logs:{capture_output}:{tail}")
        stream = FakeLogStream(returncode=self.foreground_returncode)
        self.log_streams.append(stream)
        return stream


class FakeClient:
    def __init__(
        self,
        *,
        health: Iterable[bool] = (True,),
        completions: Iterable[HttpResult] = (),
    ) -> None:
        self._health = list(health)
        self._completions = list(completions)
        self.health_timeouts: list[float] = []
        self.completion_payloads: list[Mapping[str, object]] = []

    def health(self, *, timeout: float) -> bool:
        self.health_timeouts.append(timeout)
        if self._health:
            return self._health.pop(0)
        return False

    def completion(
        self,
        payload: Mapping[str, object],
        *,
        timeout: float,
    ) -> HttpResult:
        self.completion_payloads.append(payload)
        if self._completions:
            return self._completions.pop(0)
        return HttpResult(200, '{"id":"ok"}')


class FakeReporter:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self.plans: list[LaunchPlan] = []
        self.default_profiles: list[VariantProfile] = []

    def show_plan(self, plan: LaunchPlan) -> None:
        self.plans.append(plan)

    def show_defaults(self, profiles: Iterable[VariantProfile]) -> None:
        self.default_profiles.extend(profiles)

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def success(self, message: str) -> None:
        self.messages.append(("success", message))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))

    def container_log(self, line: str) -> None:
        self.messages.append(("log", line))

    def startup_logs(self, logs: str) -> None:
        self.messages.append(("startup-logs", logs))


class FakeSecretProvider:
    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.calls = 0

    def get_hf_token(self) -> str | None:
        self.calls += 1
        return self.token


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _launcher(
    runtime: FakeRuntime,
    client: FakeClient,
    reporter: FakeReporter,
    *,
    token: str | None = None,
    secret_provider: FakeSecretProvider | None = None,
    clock: FakeClock | None = None,
) -> Launcher:
    provider = FakeSecretProvider(token) if secret_provider is None else secret_provider
    return Launcher(
        runtime=runtime,
        client=client,
        reporter=reporter,
        secret_provider=provider,
        clock=clock,
    )


def test_health_timeout_uses_real_monotonic_deadline(make_plan):
    plan = make_plan(
        no_warmup=True,
        env_overrides={"VLLM_READY_TIMEOUT": "3"},
    )
    runtime = FakeRuntime()
    client = FakeClient(health=(False, False, False, False))
    reporter = FakeReporter()
    clock = FakeClock()

    result = wait_for_health(
        plan,
        runtime=runtime,
        client=client,
        reporter=reporter,
        clock=clock,
    )

    assert result.ok is False
    assert "timed out after 3s" in (result.reason or "")
    assert clock.now == 3
    assert client.health_timeouts == [2.0, 2.0, 1.0]
    assert clock.sleeps == [1.0, 1.0, 1.0]


def test_health_reports_container_exit(make_plan):
    plan = make_plan(env_overrides={"VLLM_READY_TIMEOUT": "3"})
    runtime = FakeRuntime(running=False)

    result = wait_for_health(
        plan,
        runtime=runtime,
        client=FakeClient(health=(False,)),
        reporter=FakeReporter(),
        clock=FakeClock(),
    )

    assert result.ok is False
    assert result.reason == "container exited before becoming ready"


def test_missing_required_token_is_rejected_before_any_docker_action(make_plan):
    plan = make_plan("qwen36-fp8", detach=True)
    runtime = FakeRuntime(exists=True)

    with pytest.raises(LaunchError, match="HF token required"):
        _launcher(runtime, FakeClient(), FakeReporter()).launch(plan)

    assert runtime.events == []
    assert runtime.exists is True


def test_optional_token_is_forwarded_for_hosted_variant(make_plan):
    plan = make_plan(
        "gemma4-nvfp4",
        detach=True,
        no_warmup=True,
        no_smoke_check=True,
    )
    runtime = FakeRuntime()
    provider = FakeSecretProvider("seed-token")

    code = _launcher(
        runtime,
        FakeClient(),
        FakeReporter(),
        secret_provider=provider,
    ).launch(plan)

    assert code == 0
    assert provider.calls == 1
    assert runtime.started_hf_token == "seed-token"


def test_optional_token_can_be_absent_for_hosted_variant(make_plan):
    plan = make_plan(
        "ornith-nvfp4",
        detach=True,
        no_warmup=True,
        no_smoke_check=True,
    )
    runtime = FakeRuntime()

    code = _launcher(runtime, FakeClient(), FakeReporter()).launch(plan)

    assert code == 0
    assert runtime.started_hf_token is None


def test_preloaded_model_bypasses_optional_token(make_plan, tmp_path):
    root = tmp_path / "models"
    (root / "Gemma-4-26B-A4B-NVFP4").mkdir(parents=True)
    plan = make_plan(
        "gemma4-nvfp4",
        detach=True,
        no_warmup=True,
        no_smoke_check=True,
        use_preloaded_models=True,
        preloaded_models_dir=str(root),
    )
    runtime = FakeRuntime()
    provider = FakeSecretProvider("seed-token")

    _launcher(
        runtime,
        FakeClient(),
        FakeReporter(),
        secret_provider=provider,
    ).launch(plan)

    assert provider.calls == 0
    assert runtime.started_hf_token is None


def test_missing_preloaded_model_warning_is_reported(make_plan, tmp_path):
    plan = make_plan(
        detach=True,
        no_warmup=True,
        no_smoke_check=True,
        use_preloaded_models=True,
        preloaded_models_dir=str(tmp_path / "missing"),
    )
    reporter = FakeReporter()

    _launcher(FakeRuntime(), FakeClient(), reporter).launch(plan)

    warnings = [message for level, message in reporter.messages if level == "warning"]
    assert any("Preloaded model not found" in message for message in warnings)


def test_prepare_completes_before_managed_container_is_replaced(make_plan):
    plan = make_plan(
        detach=True,
        no_warmup=True,
        no_smoke_check=True,
    )
    runtime = FakeRuntime(exists=True, managed=True, running=True)

    code = _launcher(runtime, FakeClient(), FakeReporter()).launch(plan)

    assert code == 0
    assert runtime.events.index("prepare") < runtime.events.index("remove")
    assert runtime.events.count("remove") == 1
    assert runtime.exists is True
    assert runtime.running is True


def test_prepare_failure_leaves_existing_container_untouched(make_plan):
    plan = make_plan(detach=True, no_warmup=True, no_smoke_check=True)
    runtime = FakeRuntime(
        exists=True,
        prepare_error=RuntimeError("image unavailable"),
    )

    with pytest.raises(RuntimeError, match="image unavailable"):
        _launcher(runtime, FakeClient(), FakeReporter()).launch(plan)

    assert runtime.events == ["prepare"]
    assert runtime.exists is True


def test_unmanaged_container_is_never_removed(make_plan):
    plan = make_plan(detach=True, no_warmup=True, no_smoke_check=True)
    runtime = FakeRuntime(exists=True, managed=False)

    with pytest.raises(LaunchError, match="unmanaged container"):
        _launcher(runtime, FakeClient(), FakeReporter()).launch(plan)

    assert "remove" not in runtime.events
    assert "start" not in runtime.events
    assert runtime.exists is True


def test_failed_smoke_check_fails_launch_and_cleans_container(make_plan):
    plan = make_plan(detach=True, no_warmup=True)
    runtime = FakeRuntime()
    client = FakeClient(
        completions=(HttpResult(503, '{"error":"loading"}', "unavailable"),)
    )

    with pytest.raises(ReadinessError, match="smoke check failed"):
        _launcher(runtime, client, FakeReporter()).launch(plan)

    assert runtime.exists is False
    assert "stop" in runtime.events
    assert "remove" in runtime.events


def test_failed_warmup_fails_launch_after_all_attempts(make_plan):
    plan = make_plan(
        detach=True,
        no_smoke_check=True,
        env_overrides={"VLLM_WARMUP_REQUESTS": "2"},
    )
    runtime = FakeRuntime()
    client = FakeClient(
        completions=(
            HttpResult(500, "failed", "server error"),
            HttpResult(200, '{"id":"ok"}'),
        )
    )

    with pytest.raises(ReadinessError, match="warmup 1 failed"):
        _launcher(runtime, client, FakeReporter()).launch(plan)

    assert len(client.completion_payloads) == 2
    assert runtime.exists is False


def test_failed_start_cleans_residual_managed_container(make_plan):
    plan = make_plan(detach=True, no_warmup=True, no_smoke_check=True)
    runtime = FakeRuntime(start_error_after_create=RuntimeError("lost docker response"))

    with pytest.raises(RuntimeError, match="lost docker response"):
        _launcher(runtime, FakeClient(), FakeReporter()).launch(plan)

    assert runtime.exists is False
    assert "remove" in runtime.events


def test_foreground_exit_code_is_returned_and_container_is_cleaned(make_plan):
    plan = make_plan(no_warmup=True, no_smoke_check=True)
    runtime = FakeRuntime(foreground_returncode=7)

    code = _launcher(runtime, FakeClient(), FakeReporter()).launch(plan)

    assert code == 7
    assert runtime.exists is False
    assert runtime.log_streams[-1].closed is True


def test_warmup_and_smoke_write_artifacts(make_plan):
    plan = make_plan(env_overrides={"VLLM_WARMUP_REQUESTS": "1"})
    plan.artifact_dir.mkdir(parents=True)
    client = FakeClient(
        completions=(
            HttpResult(200, '{"id":"warm"}'),
            HttpResult(200, '{"id":"smoke"}'),
        )
    )
    reporter = FakeReporter()

    warmup = run_warmup(plan, client=client, reporter=reporter)
    smoke = smoke_check(plan, client=client, reporter=reporter)

    assert warmup.ok is True
    assert smoke.ok is True
    assert (plan.artifact_dir / f"vllm_warmup_{plan.container_name}_1.json").read_text(
        encoding="utf-8"
    ) == '{"id":"warm"}'
    assert (plan.artifact_dir / f"vllm_smoke_{plan.container_name}.json").read_text(
        encoding="utf-8"
    ) == '{"id":"smoke"}'


def test_orchestrator_show_defaults_does_not_compose_runtime():
    runtime = FakeRuntime(prepare_error=AssertionError("runtime should not be used"))
    reporter = FakeReporter()

    code = orchestrator.run(
        LaunchArgs(variant=None, show_defaults=True),
        runtime=runtime,
        reporter=reporter,
    )

    assert code == 0
    assert runtime.events == []
    assert {profile.variant for profile in reporter.default_profiles} == {
        "qwen36-fp8",
        "qwen36-nvfp4",
        "gemma4-nvfp4",
        "ornith-nvfp4",
    }


def test_orchestrator_catches_configuration_errors_before_composition():
    runtime = FakeRuntime()
    reporter = FakeReporter()

    code = orchestrator.run(
        LaunchArgs(variant="qwen36-fp8"),
        env={"VLLM_READY_TIMEOUT": "invalid"},
        runtime=runtime,
        reporter=reporter,
    )

    assert code == 1
    assert runtime.events == []
    assert reporter.messages[-1][0] == "error"
    assert "VLLM_READY_TIMEOUT" in reporter.messages[-1][1]


def test_cli_signal_handlers_restore_process_handlers():
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    with orchestrator.cli_signal_handlers():
        assert (
            signal.getsignal(signal.SIGINT)
            is orchestrator._signal_to_keyboard_interrupt
        )
        assert (
            signal.getsignal(signal.SIGTERM)
            is orchestrator._signal_to_keyboard_interrupt
        )

    assert signal.getsignal(signal.SIGINT) is previous_sigint
    assert signal.getsignal(signal.SIGTERM) is previous_sigterm
