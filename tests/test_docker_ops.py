from __future__ import annotations

from dgx_vllm_launcher import docker_ops


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


def test_run_docker_noexcept_always_returns_process_on_failure(monkeypatch):
    def fake_called(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(docker_ops.subprocess, "run", fake_called)
    proc = docker_ops.run_docker_noexcept(["docker", "bad-command"], capture_output=True)
    assert proc.returncode == 1
