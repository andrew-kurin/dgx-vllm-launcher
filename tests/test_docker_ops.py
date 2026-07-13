from __future__ import annotations

import signal
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

import pytest

from dgx_vllm_launcher import docker_ops
from dgx_vllm_launcher.docker_ops import (
    LAUNCH_INSTANCE_LABEL,
    MANAGED_LABEL,
    MANAGED_LABEL_VALUE,
    CommandResult,
    DockerCommandError,
    DockerLogStream,
    DockerRuntime,
    build_start_command,
)
from dgx_vllm_launcher.launcher import ContainerNotFoundError


def test_docker_cmd_no_sg(monkeypatch):
    monkeypatch.setattr(docker_ops, "has_sg", lambda: False)
    assert docker_ops.docker_cmd(["docker", "ps"]) == ["docker", "ps"]


def test_docker_cmd_with_sg(monkeypatch):
    monkeypatch.setattr(docker_ops, "has_sg", lambda: True)
    assert docker_ops.docker_cmd(["docker", "ps", "--format", "{{.ID}}"]) == [
        "sg",
        "docker",
        "-c",
        "docker ps --format '{{.ID}}'",
    ]


def test_has_sg_detection_is_cached(monkeypatch):
    calls = 0

    def fake_which(_executable):
        nonlocal calls
        calls += 1
        return "/usr/bin/sg"

    monkeypatch.setattr(docker_ops.shutil, "which", fake_which)
    docker_ops.has_sg.cache_clear()

    try:
        assert docker_ops.has_sg() is True
        assert docker_ops.has_sg() is True
    finally:
        docker_ops.has_sg.cache_clear()

    assert calls == 1


def test_build_start_command_contains_complete_plan_and_management_label(
    make_plan,
    tmp_path,
):
    preloaded_root = tmp_path / "models"
    (preloaded_root / "Qwen3.6-35B-A3B-NVFP4").mkdir(parents=True)
    plan = make_plan(
        reasoning=True,
        linear_backend="flashinfer",
        restart_policy="unless-stopped",
        use_preloaded_models=True,
        preloaded_models_dir=str(preloaded_root),
        env_overrides={"VLLM_HOST_PORT": "9000"},
    )

    command = build_start_command(plan, launch_id="launch-123")
    joined = " ".join(command)

    assert "127.0.0.1:9000:8000" in command
    assert MANAGED_LABEL_VALUE != "true"
    assert f"{MANAGED_LABEL}={MANAGED_LABEL_VALUE}" in command
    assert f"{MANAGED_LABEL}=true" not in command
    assert f"{LAUNCH_INSTANCE_LABEL}=launch-123" in command
    assert "--restart" in command and "unless-stopped" in command
    assert "--moe-backend" in command and "marlin" in command
    assert "--linear-backend" in command and "flashinfer" in command
    assert ":/model:ro" in joined


def test_build_start_command_formats_ipv6_bind_address_for_docker(make_plan):
    plan = make_plan(env_overrides={"VLLM_BIND_ADDRESS": "::1"})

    command = build_start_command(plan, launch_id="launch-123")

    assert "[::1]:8000:8000" in command


def test_nemotron_omni_installs_pinned_audio_extras_before_vllm(make_plan):
    plan = make_plan("nemotron3-nano-omni-nvfp4")

    command = build_start_command(plan, launch_id="launch-123")
    image_index = command.index(plan.image)

    assert command[command.index("--entrypoint") + 1] == "/bin/bash"
    assert command[image_index + 1] == "-c"
    setup_command = command[image_index + 2]
    assert "python3 -m pip install" in setup_command
    assert "--no-deps" in setup_command
    assert "--no-cache-dir" in setup_command
    assert "--target /root/.cache/vllm/python-packages/" in setup_command
    assert "importlib.metadata" in setup_command
    assert "export PYTHONPATH=/root/.cache/vllm/python-packages/" in setup_command
    for package in plan.startup_python_packages:
        assert package in setup_command
    assert 'exec vllm serve "$@"' in setup_command
    assert command[image_index + 3] == "vllm"
    assert command[image_index + 4] == plan.model


def test_hf_token_is_referenced_but_never_embedded_in_command(make_plan):
    plan = make_plan("qwen36-fp8")

    command = build_start_command(
        plan,
        launch_id="launch-123",
        include_hf_token=True,
    )

    token_index = command.index("HF_TOKEN")
    assert command[token_index - 1] == "-e"
    assert not any(part.startswith("HF_TOKEN=") for part in command)


def test_runtime_passes_hf_token_only_in_child_environment(monkeypatch, make_plan):
    plan = make_plan("qwen36-fp8")
    captured: dict[str, Any] = {}

    def fake_run_docker(args, **kwargs):
        captured["args"] = list(args)
        captured["env"] = kwargs.get("env")
        return CommandResult(0, "container-id\n", "", tuple(args))

    monkeypatch.setattr(docker_ops, "run_docker", fake_run_docker)
    runtime = DockerRuntime(process_env={"PATH": "/usr/bin"})

    container_id = runtime.start(
        plan,
        hf_token="super-secret-token",
        launch_id="launch-123",
    )

    assert container_id == "container-id"
    assert "super-secret-token" not in " ".join(captured["args"])
    child_env = captured["env"]
    assert isinstance(child_env, Mapping)
    assert child_env["HF_TOKEN"] == "super-secret-token"


def test_optional_hf_token_is_only_added_when_available(make_plan):
    plan = make_plan("gemma4-nvfp4")

    without_token = build_start_command(plan, launch_id="launch-123")
    with_token = build_start_command(
        plan,
        launch_id="launch-123",
        include_hf_token=True,
    )

    assert "HF_TOKEN" not in without_token
    assert "HF_TOKEN" in with_token
    assert not any(part.startswith("HF_TOKEN=") for part in with_token)


def test_preloaded_model_command_does_not_accept_hf_token(make_plan, tmp_path):
    root = tmp_path / "models"
    (root / "Gemma-4-26B-A4B-NVFP4").mkdir(parents=True)
    plan = make_plan(
        "gemma4-nvfp4",
        use_preloaded_models=True,
        preloaded_models_dir=str(root),
    )

    assert plan.inject_hf_token is False
    with pytest.raises(ValueError, match="does not accept"):
        build_start_command(
            plan,
            launch_id="launch-123",
            include_hf_token=True,
        )


def test_docker_errors_redact_secret_assignments(monkeypatch):
    monkeypatch.setattr(docker_ops, "has_sg", lambda: False)

    def failed_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["docker", "run"],
            returncode=1,
            stdout="",
            stderr="launch failed",
        )

    monkeypatch.setattr(docker_ops.subprocess, "run", failed_run)

    with pytest.raises(DockerCommandError) as exc_info:
        docker_ops.run_docker(
            ["docker", "run", "-e", "HF_TOKEN=visible-secret", "image"]
        )

    message = str(exc_info.value)
    assert "visible-secret" not in message
    assert "HF_TOKEN=<redacted>" in message
    assert "launch failed" in message


def test_prepare_pulls_image_before_launch_when_missing(monkeypatch, make_plan):
    plan = make_plan()
    commands: list[list[str]] = []

    def fake_run_docker(args, **kwargs):
        command = list(args)
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return CommandResult(
                1,
                "",
                f"Error: No such image: {plan.image}",
                tuple(command),
            )
        return CommandResult(0, "ok", "", tuple(command))

    monkeypatch.setattr(docker_ops, "run_docker", fake_run_docker)

    DockerRuntime().prepare(plan)

    assert commands[0][1] == "version"
    assert commands[1][1:3] == ["image", "inspect"]
    assert commands[2][1:3] == ["pull", plan.image]


@pytest.mark.parametrize(
    ("version", "bind_address", "expects_warning"),
    [
        ("27.5.1", "127.0.0.1", True),
        ("v27.5.1", "::1", True),
        ("28.0.0", "127.0.0.1", False),
        ("27.5.1", "0.0.0.0", False),
        ("development", "127.0.0.1", False),
    ],
)
def test_prepare_warns_for_loopback_binding_on_older_docker_engines(
    monkeypatch,
    make_plan,
    version,
    bind_address,
    expects_warning,
):
    plan = make_plan(env_overrides={"VLLM_BIND_ADDRESS": bind_address})

    def fake_run_docker(args, **kwargs):
        command = list(args)
        stdout = version if command[1] == "version" else "ok"
        return CommandResult(0, stdout, "", tuple(command))

    monkeypatch.setattr(docker_ops, "run_docker", fake_run_docker)

    warnings = DockerRuntime().prepare(plan)

    assert bool(warnings) is expects_warning
    if expects_warning:
        assert "SECURITY WARNING" in warnings[0]
        assert version in warnings[0]


def test_container_exists_distinguishes_absence_from_docker_failure(monkeypatch):
    def missing_container(args, **kwargs):
        return CommandResult(
            1,
            "",
            "Error: No such object: missing",
            tuple(args),
        )

    monkeypatch.setattr(docker_ops, "run_docker", missing_container)
    assert DockerRuntime().container_exists("missing") is False

    def daemon_failure(args, **kwargs):
        return CommandResult(
            1,
            "",
            "Cannot connect to the Docker daemon",
            tuple(args),
        )

    monkeypatch.setattr(docker_ops, "run_docker", daemon_failure)
    with pytest.raises(DockerCommandError, match="Cannot connect"):
        DockerRuntime().container_exists("missing")


def test_container_id_distinguishes_identity_from_absence(monkeypatch):
    results = iter(
        (
            CommandResult(0, "full-container-id\n", "", ("docker", "inspect")),
            CommandResult(
                1,
                "",
                "Error: No such container: missing",
                ("docker", "inspect"),
            ),
        )
    )
    monkeypatch.setattr(
        docker_ops, "run_docker", lambda *_args, **_kwargs: next(results)
    )
    runtime = DockerRuntime()

    assert runtime.container_id("named") == "full-container-id"
    assert runtime.container_id("missing") is None


@pytest.mark.parametrize(
    ("labels", "launch_id", "expected"),
    [
        ({MANAGED_LABEL: MANAGED_LABEL_VALUE}, None, True),
        ({MANAGED_LABEL: "true"}, None, True),
        ({MANAGED_LABEL: "false"}, None, False),
        (
            {
                MANAGED_LABEL: MANAGED_LABEL_VALUE,
                LAUNCH_INSTANCE_LABEL: "launch-123",
            },
            "launch-123",
            True,
        ),
        (
            {MANAGED_LABEL: "true", LAUNCH_INSTANCE_LABEL: "launch-123"},
            "launch-123",
            False,
        ),
        (
            {MANAGED_LABEL: MANAGED_LABEL_VALUE, LAUNCH_INSTANCE_LABEL: "other"},
            "launch-123",
            False,
        ),
        (None, None, False),
    ],
)
def test_container_is_managed_parses_labels_and_launch_identity(
    monkeypatch,
    labels,
    launch_id,
    expected,
):
    monkeypatch.setattr(
        docker_ops,
        "run_docker",
        lambda args, **_kwargs: CommandResult(
            0,
            docker_ops.json.dumps(labels),
            "",
            tuple(args),
        ),
    )

    assert (
        DockerRuntime().container_is_managed(
            "container-id",
            launch_id=launch_id,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"Running": True, "ExitCode": 0}, None),
        ({"Running": False, "ExitCode": 17}, 17),
    ],
)
def test_container_exit_code_uses_actual_docker_state(
    monkeypatch,
    state,
    expected,
):
    monkeypatch.setattr(
        docker_ops,
        "run_docker",
        lambda args, **_kwargs: CommandResult(
            0,
            docker_ops.json.dumps(state),
            "",
            tuple(args),
        ),
    )

    assert DockerRuntime().container_exit_code("container-id", timeout=10) is expected


def test_container_state_inspection_distinguishes_absence_from_transient_failure(
    monkeypatch,
):
    def missing_container(*_args, **_kwargs):
        raise DockerCommandError(
            "Docker command failed",
            command=["docker", "inspect", "missing"],
            returncode=1,
            stderr="Error: No such object: missing",
        )

    monkeypatch.setattr(docker_ops, "run_docker", missing_container)

    with pytest.raises(ContainerNotFoundError, match="no longer exists"):
        DockerRuntime().container_running("missing", timeout=10)


def test_container_label_inspection_classifies_confirmed_absence(monkeypatch):
    monkeypatch.setattr(
        docker_ops,
        "run_docker",
        lambda args, **_kwargs: CommandResult(
            1,
            "",
            "Error: No such object: missing",
            tuple(args),
        ),
    )

    with pytest.raises(ContainerNotFoundError, match="no longer exists"):
        DockerRuntime().container_is_managed("missing")


@pytest.mark.parametrize("operation", ["stop", "remove"])
def test_container_mutations_classify_confirmed_absence(monkeypatch, operation):
    monkeypatch.setattr(
        docker_ops,
        "run_docker",
        lambda args, **_kwargs: CommandResult(
            1,
            "",
            "Error: No such container: missing",
            tuple(args),
        ),
    )

    with pytest.raises(ContainerNotFoundError, match="no longer exists"):
        getattr(DockerRuntime(), operation)("missing")


def test_logs_include_both_output_streams(monkeypatch):
    def fake_run_docker(args, **kwargs):
        return CommandResult(0, "stdout log", "stderr log", tuple(args))

    monkeypatch.setattr(docker_ops, "run_docker", fake_run_docker)

    assert DockerRuntime().logs("container") == "stdout log\nstderr log"


def test_log_stream_close_tolerates_process_that_survives_sigkill(monkeypatch):
    class StuckProcess:
        pid = 1234
        stdout = None

        def poll(self):
            return None

        def wait(self, timeout: float | None = None):
            raise subprocess.TimeoutExpired("docker logs", timeout or 0)

    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        docker_ops.os,
        "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
    )

    DockerLogStream(StuckProcess()).close()  # type: ignore[arg-type]

    assert signals == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]


def test_log_stream_close_escalates_for_real_process_ignoring_sigterm():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); "
                "time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"

        DockerLogStream(process).close()

        assert process.returncode == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
