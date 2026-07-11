from __future__ import annotations

import subprocess
from collections.abc import Mapping
from typing import Any

import pytest

from dgx_vllm_launcher import docker_ops
from dgx_vllm_launcher.docker_ops import (
    MANAGED_LABEL,
    CommandResult,
    DockerCommandError,
    DockerRuntime,
    build_start_command,
)


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

    command = build_start_command(plan)
    joined = " ".join(command)

    assert "9000:8000" in command
    assert f"{MANAGED_LABEL}=true" in command
    assert "--restart" in command and "unless-stopped" in command
    assert "--moe-backend" in command and "marlin" in command
    assert "--linear-backend" in command and "flashinfer" in command
    assert ":/model:ro" in joined


def test_hf_token_is_referenced_but_never_embedded_in_command(make_plan):
    plan = make_plan("qwen36-fp8")

    command = build_start_command(plan, include_hf_token=True)

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

    container_id = runtime.start(plan, hf_token="super-secret-token")

    assert container_id == "container-id"
    assert "super-secret-token" not in " ".join(captured["args"])
    child_env = captured["env"]
    assert isinstance(child_env, Mapping)
    assert child_env["HF_TOKEN"] == "super-secret-token"


def test_optional_hf_token_is_only_added_when_available(make_plan):
    plan = make_plan("gemma4-nvfp4")

    without_token = build_start_command(plan)
    with_token = build_start_command(plan, include_hf_token=True)

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
        build_start_command(plan, include_hf_token=True)


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


def test_logs_include_both_output_streams(monkeypatch):
    def fake_run_docker(args, **kwargs):
        return CommandResult(0, "stdout log", "stderr log", tuple(args))

    monkeypatch.setattr(docker_ops, "run_docker", fake_run_docker)

    assert DockerRuntime().logs("container") == "stdout log\nstderr log"
