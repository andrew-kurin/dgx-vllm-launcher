from __future__ import annotations

import subprocess
import sys

import pytest

from dgx_vllm_launcher import check


def test_check_returns_failing_tool_code_without_running_later_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    returncodes = iter((0, 7))

    def fake_run(
        command: list[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, next(returncodes))

    monkeypatch.setattr(check.subprocess, "run", fake_run)

    assert check.main() == 7
    assert calls == [
        [sys.executable, "-m", "ruff", "check", "dgx_vllm_launcher", "tests"],
        [sys.executable, "-m", "pyright"],
    ]


def test_check_returns_zero_after_all_tools_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(check.subprocess, "run", fake_run)

    assert check.main() == 0
    assert [command[2] for command in calls] == ["ruff", "pyright", "pytest"]
