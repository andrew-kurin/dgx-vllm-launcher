from __future__ import annotations

import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import dgx_vllm_launcher.orchestrator as orchestrator
from dgx_vllm_launcher.docker_ops import DockerCommandError, DockerRuntime
from dgx_vllm_launcher.launcher import LaunchError
from dgx_vllm_launcher.plan import LaunchArgs
from dgx_vllm_launcher.presentation import RichReporter
from dgx_vllm_launcher.secrets import HuggingFaceTokenProvider

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(sys.executable).parent
LAUNCH_COMMANDS = (
    (str(SCRIPTS_DIR / "dgx-vllm-launcher"),),
    (str(SCRIPTS_DIR / "dgxvllm"),),
    (str(SCRIPTS_DIR / "dvl"),),
    (sys.executable, "-m", "dgx_vllm_launcher"),
)


def _launcher_env() -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("VLLM_", "HF_", "HUGGING_FACE_"))
    }
    env["NO_COLOR"] = "1"
    return env


def _plan_env(tmp_path: Path) -> dict[str, str]:
    return {
        "VLLM_CACHE_DIR": str(tmp_path / "vllm-cache"),
        "VLLM_HF_CACHE_DIR": str(tmp_path / "hf-cache"),
        "VLLM_ARTIFACT_DIR": str(tmp_path / "artifacts"),
        "VLLM_PRELOADED_MODELS_DIR": str(tmp_path / "models"),
    }


@pytest.mark.parametrize(
    "command",
    LAUNCH_COMMANDS,
    ids=("full", "compact", "short", "module"),
)
def test_installed_launcher_entrypoints_render_defaults(
    command: tuple[str, ...],
) -> None:
    completed = subprocess.run(
        [*command, "--show-defaults"],
        cwd=PROJECT_ROOT,
        env=_launcher_env(),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "Default launch configuration" in output
    assert "qwen36-fp8" in output


def test_launcher_help_describes_configured_profiles() -> None:
    completed = subprocess.run(
        [str(SCRIPTS_DIR / "dvl"), "--help"],
        cwd=PROJECT_ROOT,
        env=_launcher_env(),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "Print configured per-variant launch settings and exit" in output


def test_config_error_entrypoint_exits_cleanly_without_docker() -> None:
    env = _launcher_env()
    env["PATH"] = ""
    env["VLLM_HOST_PORT"] = "not-an-integer"

    completed = subprocess.run(
        [str(SCRIPTS_DIR / "dvl"), "qwen36-fp8"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 1
    assert "Error: VLLM_HOST_PORT must be an integer" in output
    assert "Traceback" not in output


def test_run_composes_default_dependencies_without_invoking_docker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    client = object()

    def client_factory(base_url: str) -> object:
        captured["base_url"] = base_url
        return client

    class CapturingLauncher:
        def __init__(self, **dependencies: Any) -> None:
            captured.update(dependencies)

        def launch(self, plan: object) -> int:
            captured["plan"] = plan
            return 23

    monkeypatch.setattr(orchestrator, "VllmClient", client_factory)
    monkeypatch.setattr(orchestrator, "Launcher", CapturingLauncher)
    env = _plan_env(tmp_path)
    env.update(
        {
            "HF_TOKEN": "token-from-explicit-env",
            "VLLM_HOST_PORT": "9012",
        }
    )

    code = orchestrator.run(LaunchArgs(variant="qwen36-fp8"), env=env)

    assert code == 23
    assert isinstance(captured["runtime"], DockerRuntime)
    assert isinstance(captured["reporter"], RichReporter)
    assert isinstance(captured["secret_provider"], HuggingFaceTokenProvider)
    assert captured["secret_provider"].get_hf_token() == "token-from-explicit-env"
    assert captured["client"] is client
    assert captured["base_url"] == "http://127.0.0.1:9012"
    assert captured["clock"] is None
    assert captured["plan"].base_url == "http://127.0.0.1:9012"


def test_run_does_not_hide_unexpected_launcher_defects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class DefectiveLauncher:
        def __init__(self, **_dependencies: Any) -> None:
            pass

        def launch(self, _plan: object) -> int:
            raise AttributeError("unexpected launcher defect")

    monkeypatch.setattr(orchestrator, "Launcher", DefectiveLauncher)
    with pytest.raises(AttributeError, match="unexpected launcher defect"):
        orchestrator.run(
            LaunchArgs(variant="qwen36-fp8"),
            env=_plan_env(tmp_path),
        )


@pytest.mark.parametrize(
    "failure",
    (
        LaunchError("expected launch failure"),
        DockerCommandError(
            "expected Docker failure",
            command=["docker", "version"],
            returncode=1,
        ),
    ),
    ids=("launch", "docker"),
)
def test_run_reports_intentional_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: Exception,
) -> None:
    class FailingLauncher:
        def __init__(self, **_dependencies: Any) -> None:
            pass

        def launch(self, _plan: object) -> int:
            raise failure

    monkeypatch.setattr(orchestrator, "Launcher", FailingLauncher)
    output = StringIO()
    reporter = RichReporter(
        Console(file=output, color_system=None, force_terminal=False)
    )

    code = orchestrator.run(
        LaunchArgs(variant="qwen36-fp8"),
        env=_plan_env(tmp_path),
        reporter=reporter,
    )

    assert code == 1
    assert output.getvalue().startswith("Error: expected")
