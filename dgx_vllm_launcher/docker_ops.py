from __future__ import annotations

import hashlib
import os
import re
import shlex
import signal
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import NoReturn

from .plan import LaunchPlan

MANAGED_LABEL = "com.andrewkurin.dgx-vllm-launcher.managed"
VARIANT_LABEL = "com.andrewkurin.dgx-vllm-launcher.variant"

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)="
    r"(?:'[^']*'|\"[^\"]*\"|[^\s]+)"
)


def redact_command(command: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        _SECRET_ASSIGNMENT.sub(r"\1=<redacted>", str(part)) for part in command
    )


class DockerCommandError(RuntimeError):
    """A Docker command failed; the stored command is always redacted."""

    def __init__(
        self,
        message: str,
        *,
        command: list[str] | tuple[str, ...],
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        self.command = redact_command(command)
        self.returncode = returncode
        self.stderr = stderr.strip()
        detail = f"{message}: {shlex.join(self.command)}"
        if returncode is not None:
            detail += f" (exit {returncode})"
        if self.stderr:
            detail += f"\n{self.stderr}"
        super().__init__(detail)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    command: tuple[str, ...]


def has_sg() -> bool:
    return shutil.which("sg") is not None


def docker_cmd(args: list[str] | tuple[str, ...]) -> list[str]:
    cmd = [str(arg) for arg in args]
    if has_sg():
        return ["sg", "docker", "-c", shlex.join(cmd)]
    return cmd


def run_docker(
    args: list[str] | tuple[str, ...],
    *,
    check: bool = True,
    capture_output: bool = True,
    input_text: str | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    command = docker_cmd(args)
    try:
        completed = subprocess.run(
            command,
            input=input_text,
            check=False,
            text=True,
            capture_output=capture_output,
            timeout=timeout,
            env=None if env is None else dict(env),
        )
    except subprocess.TimeoutExpired as exc:
        raise DockerCommandError(
            "Docker command timed out",
            command=command,
            stderr=str(exc.stderr or ""),
        ) from exc
    except OSError as exc:
        raise DockerCommandError(
            f"Could not execute Docker command: {exc}",
            command=command,
        ) from exc

    result = CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        command=redact_command(command),
    )
    if check and result.returncode != 0:
        raise DockerCommandError(
            "Docker command failed",
            command=command,
            returncode=result.returncode,
            stderr=result.stderr,
        )
    return result


def _raise_result_error(message: str, result: CommandResult) -> NoReturn:
    raise DockerCommandError(
        message,
        command=result.command,
        returncode=result.returncode,
        stderr=result.stderr,
    )


def _python_package_setup_command(image: str, packages: tuple[str, ...]) -> str:
    expected: dict[str, str] = {}
    for package in packages:
        name, separator, version = package.partition("==")
        if not separator or not name or not version:
            raise ValueError(f"startup Python package must be exactly pinned: {package!r}")
        expected[name] = version

    cache_key = hashlib.sha256(
        "\0".join((image, *packages)).encode("utf-8")
    ).hexdigest()[:16]
    target = f"/root/.cache/vllm/python-packages/{cache_key}"
    check_code = (
        "import importlib, importlib.metadata as metadata, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        f"expected = {expected!r}; "
        "assert all(metadata.version(name) == version "
        "for name, version in expected.items()); "
        "[importlib.import_module(name.replace('-', '_')) for name in expected]"
    )
    check = shlex.join(["python3", "-c", check_code, target])
    install = shlex.join(
        [
            "python3",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--root-user-action=ignore",
            "--no-cache-dir",
            "--no-deps",
            "--target",
            target,
            "--upgrade",
            *packages,
        ]
    )
    return (
        f"set -e\nif ! {check} >/dev/null 2>&1; then\n  {install}\nfi\n"
        f"export PYTHONPATH={shlex.quote(target)}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        'exec vllm serve "$@"'
    )


def build_start_command(
    plan: LaunchPlan,
    *,
    include_hf_token: bool = False,
) -> list[str]:
    """Build a Docker command without embedding secret values."""

    command = [
        "docker",
        "run",
        "-d",
        "--gpus",
        "all",
        "-p",
        f"{plan.host_port}:{plan.container_port}",
        "--name",
        plan.container_name,
        "--label",
        f"{MANAGED_LABEL}=true",
        "--label",
        f"{VARIANT_LABEL}={plan.variant}",
        "--ipc=host",
    ]

    if plan.startup_python_packages:
        command.extend(["--entrypoint", "/bin/bash"])

    if plan.restart_policy:
        command.extend(["--restart", plan.restart_policy])

    for name, value in plan.container_env:
        command.extend(["-e", f"{name}={value}"])
    if include_hf_token:
        if not plan.inject_hf_token:
            raise ValueError("this launch plan does not accept an HF token")
        command.extend(["-e", "HF_TOKEN"])

    for mount in plan.mounts:
        value = f"{mount.host_path}:{mount.container_path}"
        if mount.read_only:
            value += ":ro"
        command.extend(["-v", value])

    command.append(plan.image)
    if plan.startup_python_packages:
        command.extend(
            [
                "-c",
                _python_package_setup_command(
                    plan.image,
                    plan.startup_python_packages,
                ),
                "vllm",
            ]
        )
    command.extend([plan.model, *plan.vllm_args])
    return command


class DockerLogStream:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process

    def lines(self) -> Iterator[str]:
        if self._process.stdout is None:
            return
        for line in self._process.stdout:
            yield line.rstrip("\n")

    def wait(self) -> int:
        return self._process.wait()

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            self._process.wait(timeout=2)


class DockerRuntime:
    """Subprocess-backed container runtime used by the application layer."""

    def __init__(self, process_env: Mapping[str, str] | None = None) -> None:
        self._process_env = os.environ if process_env is None else process_env

    def prepare(self, plan: LaunchPlan) -> None:
        """Validate Docker and ensure the image exists before replacing a service."""

        run_docker(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            timeout=15,
        )
        image = run_docker(
            ["docker", "image", "inspect", plan.image],
            check=False,
            timeout=30,
        )
        if image.returncode != 0:
            if "no such image" not in image.stderr.lower():
                _raise_result_error("Could not inspect Docker image", image)
            run_docker(
                ["docker", "pull", plan.image],
                capture_output=False,
            )

    def container_exists(self, name: str) -> bool:
        result = run_docker(
            ["docker", "inspect", name],
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            return True
        missing_markers = ("no such object", "no such container")
        if any(marker in result.stderr.lower() for marker in missing_markers):
            return False
        _raise_result_error("Could not inspect Docker container", result)

    def container_is_managed(self, name: str) -> bool:
        result = run_docker(
            [
                "docker",
                "inspect",
                name,
                "--format",
                f'{{{{ index .Config.Labels "{MANAGED_LABEL}" }}}}',
            ],
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            _raise_result_error("Could not inspect Docker container labels", result)
        return result.stdout.strip() == "true"

    def container_running(self, name: str, *, timeout: float) -> bool:
        result = run_docker(
            ["docker", "inspect", name, "--format", "{{.State.Running}}"],
            timeout=max(timeout, 0.1),
        )
        return result.stdout.strip() == "true"

    def start(self, plan: LaunchPlan, *, hf_token: str | None) -> str:
        if plan.requires_hf_token and not hf_token:
            raise ValueError("an HF token is required by this launch plan")

        include_hf_token = bool(hf_token and plan.inject_hf_token)
        process_env = dict(self._process_env)
        if include_hf_token:
            assert hf_token is not None
            process_env["HF_TOKEN"] = hf_token

        command = build_start_command(
            plan,
            include_hf_token=include_hf_token,
        )
        result = run_docker(command, env=process_env)
        container_id = result.stdout.strip()
        if not container_id:
            raise DockerCommandError(
                "docker run produced no container ID",
                command=command,
            )
        return container_id

    def logs(self, name: str, *, tail: int = 200) -> str:
        result = run_docker(
            ["docker", "logs", "--tail", str(tail), name],
            check=False,
        )
        if result.returncode != 0:
            return result.stderr.strip() or "(no logs available)"
        streams = [
            stream.strip()
            for stream in (result.stdout, result.stderr)
            if stream.strip()
        ]
        return "\n".join(streams) or "(no logs available)"

    def stop(self, name: str) -> None:
        run_docker(["docker", "stop", name])

    def remove(self, name: str) -> None:
        run_docker(["docker", "rm", name])

    def open_logs(
        self,
        name: str,
        *,
        capture_output: bool,
        tail: int | None = None,
    ) -> DockerLogStream:
        args = ["docker", "logs", "-f"]
        if tail is not None:
            args.extend(["--tail", str(tail)])
        args.append(name)
        command = docker_cmd(args)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.STDOUT if capture_output else None,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            raise DockerCommandError(
                f"Could not stream Docker logs: {exc}",
                command=command,
            ) from exc
        return DockerLogStream(process)
