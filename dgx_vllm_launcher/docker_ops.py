from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass


class DockerCommandError(RuntimeError):
    """Raised when a docker command fails."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    command: list[str]

    @classmethod
    def from_completed(cls, completed: subprocess.CompletedProcess[str]) -> "CommandResult":
        return cls(
            returncode=completed.returncode,
            stdout=(completed.stdout or ""),
            stderr=(completed.stderr or ""),
            command=completed.args if isinstance(completed.args, list) else [str(completed.args)],
        )


def has_sg() -> bool:
    return shutil.which("sg") is not None


def docker_cmd(args: list[str]) -> list[str]:
    cmd = [str(a) for a in args]
    if has_sg():
        return ["sg", "docker", "-c", shlex.join(cmd)]
    return cmd


def run_docker(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    input_text: str | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    cmd = docker_cmd(args)
    try:
        return subprocess.run(
            cmd,
            input=input_text,
            check=check,
            text=True,
            capture_output=capture_output,
            timeout=timeout,
        )
    except Exception as exc:
        raise DockerCommandError(str(exc)) from exc


def run_docker_noexcept(args: list[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    try:
        return run_docker(args, check=False, capture_output=capture_output)
    except DockerCommandError:
        return subprocess.CompletedProcess(args=docker_cmd(args), returncode=1, stdout="", stderr="")


def container_running(name: str) -> bool:
    proc = run_docker_noexcept(["docker", "inspect", name, "--format", "{{.State.Running}}"], capture_output=True)
    if proc.returncode != 0:
        return False
    return proc.stdout.strip() == "true"


def container_logs(name: str, tail: int = 200) -> str:
    proc = run_docker_noexcept(["docker", "logs", "--tail", str(tail), name], capture_output=True)
    if proc.returncode != 0:
        return "(no logs available)"
    return (proc.stdout or "").strip()


def stop_container(name: str) -> None:
    run_docker_noexcept(["docker", "stop", name], capture_output=True)


def remove_container(name: str) -> None:
    run_docker_noexcept(["docker", "rm", name], capture_output=True)


def inspect_container(name: str) -> subprocess.CompletedProcess:
    return run_docker_noexcept(["docker", "inspect", name], capture_output=True)


def stream_container_logs(name: str, *, capture_output: bool = False, tail: int | None = None) -> subprocess.Popen[str]:
    args = ["docker", "logs", "-f"]
    if tail is not None:
        args.extend(["--tail", str(tail)])
    args.append(name)
    cmd = docker_cmd(args)
    if capture_output:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    return subprocess.Popen(cmd, text=True)
