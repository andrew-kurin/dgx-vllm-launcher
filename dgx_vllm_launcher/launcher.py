from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .http_ops import HttpResult
from .plan import LaunchPlan
from .presentation import BackgroundLogTailer, Reporter

HEALTH_REQUEST_TIMEOUT_SECONDS = 2.0
HEALTH_POLL_INTERVAL_SECONDS = 1.0
COMPLETION_TIMEOUT_SECONDS = 120.0


class LaunchError(RuntimeError):
    """Raised when a launch cannot safely complete."""


class ReadinessError(LaunchError):
    """Raised when a started service fails a required readiness check."""


class RuntimeLogStream(Protocol):
    def lines(self) -> Iterable[str]: ...

    def wait(self) -> int: ...

    def close(self) -> None: ...


class ContainerRuntime(Protocol):
    def prepare(self, plan: LaunchPlan) -> None: ...

    def container_exists(self, name: str) -> bool: ...

    def container_is_managed(self, name: str) -> bool: ...

    def container_running(self, name: str, *, timeout: float) -> bool: ...

    def start(self, plan: LaunchPlan, *, hf_token: str | None) -> str: ...

    def logs(self, name: str, *, tail: int = 200) -> str: ...

    def stop(self, name: str) -> None: ...

    def remove(self, name: str) -> None: ...

    def open_logs(
        self,
        name: str,
        *,
        capture_output: bool,
        tail: int | None = None,
    ) -> RuntimeLogStream: ...


class VllmService(Protocol):
    def health(self, *, timeout: float) -> bool: ...

    def completion(
        self,
        payload: Mapping[str, object],
        *,
        timeout: float,
    ) -> HttpResult: ...


class SecretProvider(Protocol):
    def get_hf_token(self) -> str | None: ...


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class HealthResult:
    ok: bool
    reason: str | None = None


@dataclass(frozen=True)
class CheckResult:
    name: str
    attempts: int
    failures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures


class Launcher:
    def __init__(
        self,
        *,
        runtime: ContainerRuntime,
        client: VllmService,
        reporter: Reporter,
        secret_provider: SecretProvider,
        clock: Clock | None = None,
    ) -> None:
        self._runtime = runtime
        self._client = client
        self._reporter = reporter
        self._secret_provider = secret_provider
        self._clock = SystemClock() if clock is None else clock

    def launch(self, plan: LaunchPlan) -> int:
        """Launch one plan, cleaning up unless detached startup fully succeeds."""

        hf_token = self._resolve_hf_token(plan)
        self._prepare_paths(plan)
        self._reporter.show_plan(plan)
        for warning in plan.warnings:
            self._reporter.warning(warning)
        self._reporter.info("Validating Docker and preparing the image...")
        self._runtime.prepare(plan)
        self._replace_existing_container(plan.container_name)

        launch_attempted = True
        cleanup_container = True
        try:
            self._reporter.success(plan.startup_message)
            container_id = self._runtime.start(plan, hf_token=hf_token)
            self._reporter.success(
                f"Started container {container_id} as {plan.container_name}"
            )

            health = wait_for_health(
                plan,
                runtime=self._runtime,
                client=self._client,
                reporter=self._reporter,
                clock=self._clock,
            )
            if not health.ok:
                self._reporter.startup_logs(
                    self._runtime.logs(plan.container_name, tail=200)
                )
                raise ReadinessError(health.reason or "health check failed")

            self._reporter.success(
                "Service is healthy. Running required warmup and smoke checks."
            )
            warmup = run_warmup(
                plan,
                client=self._client,
                reporter=self._reporter,
            )
            if not warmup.ok:
                raise ReadinessError("; ".join(warmup.failures))

            if plan.run_smoke_check:
                smoke = smoke_check(
                    plan,
                    client=self._client,
                    reporter=self._reporter,
                )
                if not smoke.ok:
                    raise ReadinessError("; ".join(smoke.failures))

            if plan.detach:
                cleanup_container = False
                self._reporter.success(
                    "Startup checks passed; container is running in detached mode."
                )
                self._reporter.info(
                    f"Tail logs with: docker logs -f {plan.container_name}"
                )
                self._reporter.info(
                    f"Stop container with: docker stop {plan.container_name}"
                )
                return 0

            self._reporter.info(
                "Streaming logs. Press Ctrl-C to stop; the container will be removed."
            )
            stream = self._runtime.open_logs(
                plan.container_name,
                capture_output=False,
                tail=20,
            )
            try:
                return stream.wait()
            finally:
                stream.close()
        except KeyboardInterrupt:
            self._reporter.warning("Interrupted. Exiting...")
            return 130
        finally:
            if launch_attempted and cleanup_container:
                self._cleanup_managed_container(plan.container_name)

    def _resolve_hf_token(self, plan: LaunchPlan) -> str | None:
        if not plan.inject_hf_token:
            return None
        token = self._secret_provider.get_hf_token()
        if plan.requires_hf_token and not token:
            raise LaunchError(
                "HF token required; set HF_TOKEN or run `huggingface-cli login`"
            )
        return token

    def _prepare_paths(self, plan: LaunchPlan) -> None:
        writable_paths = [
            mount.host_path for mount in plan.mounts if not mount.read_only
        ]
        writable_paths.append(plan.artifact_dir)
        for path in writable_paths:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise LaunchError(f"could not prepare directory {path}: {exc}") from exc
            if not path.is_dir():
                raise LaunchError(f"configured path is not a directory: {path}")

    def _replace_existing_container(self, name: str) -> None:
        if not self._runtime.container_exists(name):
            return
        if not self._runtime.container_is_managed(name):
            raise LaunchError(
                f"refusing to remove unmanaged container {name}; "
                "rename or remove it explicitly"
            )

        self._reporter.warning(f"Replacing managed container {name}")
        if self._runtime.container_running(name, timeout=10):
            self._runtime.stop(name)
        self._runtime.remove(name)

    def _cleanup_managed_container(self, name: str) -> None:
        try:
            if not self._runtime.container_exists(name):
                return
            if not self._runtime.container_is_managed(name):
                self._reporter.warning(
                    f"Refusing to clean up unmanaged container {name}"
                )
                return
            self._reporter.warning(f"Cleaning up container {name}")
            if self._runtime.container_running(name, timeout=10):
                self._runtime.stop(name)
            self._runtime.remove(name)
        except Exception as exc:
            self._reporter.warning(f"Container cleanup failed: {exc}")


def wait_for_health(
    plan: LaunchPlan,
    *,
    runtime: ContainerRuntime,
    client: VllmService,
    reporter: Reporter,
    clock: Clock,
) -> HealthResult:
    """Poll against a monotonic deadline rather than treating seconds as attempts."""

    reporter.info(f"Waiting up to {plan.ready_timeout_seconds}s for /health ...")
    deadline = clock.monotonic() + plan.ready_timeout_seconds
    tailer = BackgroundLogTailer(
        lambda: runtime.open_logs(
            plan.container_name,
            capture_output=True,
            tail=None,
        ),
        reporter.container_log,
    )
    try:
        tailer.start()
    except Exception as exc:
        tailer.stop()
        reporter.warning(f"Could not stream startup logs: {exc}")
        tailer = None

    try:
        while True:
            remaining = deadline - clock.monotonic()
            if remaining <= 0:
                return HealthResult(
                    ok=False,
                    reason=(
                        f"timed out after {plan.ready_timeout_seconds}s "
                        "waiting for the health endpoint"
                    ),
                )

            inspect_timeout = min(10.0, remaining)
            if not runtime.container_running(
                plan.container_name,
                timeout=inspect_timeout,
            ):
                return HealthResult(
                    ok=False,
                    reason="container exited before becoming ready",
                )

            remaining = deadline - clock.monotonic()
            if remaining <= 0:
                continue
            if client.health(timeout=min(HEALTH_REQUEST_TIMEOUT_SECONDS, remaining)):
                return HealthResult(ok=True)

            remaining = deadline - clock.monotonic()
            if remaining > 0:
                clock.sleep(min(HEALTH_POLL_INTERVAL_SECONDS, remaining))
    finally:
        if tailer is not None:
            tailer.stop()


def run_warmup(
    plan: LaunchPlan,
    *,
    client: VllmService,
    reporter: Reporter,
) -> CheckResult:
    if plan.warmup_requests == 0:
        reporter.info("Warmup skipped.")
        return CheckResult(name="warmup", attempts=0)

    reporter.info(f"Running {plan.warmup_requests} warmup completion(s)...")
    prompt = (
        "The purpose of this request is warmup only. Please ignore the content "
        "and return a short completion immediately without special formatting."
    )
    failures: list[str] = []
    for index in range(1, plan.warmup_requests + 1):
        result = client.completion(
            {
                "model": plan.served_model_name,
                "prompt": prompt,
                "max_tokens": 2,
                "temperature": 0.0,
                "return_token_ids": True,
            },
            timeout=COMPLETION_TIMEOUT_SECONDS,
        )
        if result.ok and result.body is not None:
            path = plan.artifact_dir / f"vllm_warmup_{plan.container_name}_{index}.json"
            _save_artifact(path, result.body, reporter)
            reporter.success(f"Warmup {index}/{plan.warmup_requests} completed")
        else:
            detail = result.failure_detail()
            failures.append(f"warmup {index} failed: {detail}")
            reporter.error(f"Warmup {index}/{plan.warmup_requests} failed: {detail}")

    return CheckResult(
        name="warmup",
        attempts=plan.warmup_requests,
        failures=tuple(failures),
    )


def smoke_check(
    plan: LaunchPlan,
    *,
    client: VllmService,
    reporter: Reporter,
) -> CheckResult:
    result = client.completion(
        {
            "model": plan.served_model_name,
            "prompt": "Smoke test request.",
            "max_tokens": 4,
            "temperature": 0.0,
            "return_token_ids": True,
        },
        timeout=COMPLETION_TIMEOUT_SECONDS,
    )
    if result.ok and result.body is not None:
        path = plan.artifact_dir / f"vllm_smoke_{plan.container_name}.json"
        _save_artifact(path, result.body, reporter)
        reporter.success(f"Smoke check passed. Response saved to {path}")
        for line in result.body.splitlines()[:3]:
            reporter.container_log(line)
        return CheckResult(name="smoke", attempts=1)

    detail = result.failure_detail()
    reporter.error(f"Smoke check failed: {detail}")
    return CheckResult(
        name="smoke",
        attempts=1,
        failures=(f"smoke check failed: {detail}",),
    )


def _save_artifact(path: Path, body: str, reporter: Reporter) -> None:
    try:
        path.write_text(body, encoding="utf-8")
    except OSError as exc:
        reporter.warning(f"Could not save response artifact {path}: {exc}")
