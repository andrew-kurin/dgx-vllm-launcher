from __future__ import annotations

from qwen_vllm_launcher import cli
import qwen_vllm_launcher.orchestrator as orchestrator
from qwen_vllm_launcher.orchestrator import (
    build_common_args,
    build_start_command,
    run_warmup,
    smoke_check,
    wait_for_health,
)


def test_build_common_args_without_reasoning():
    args = build_common_args("qwen36-fp8", reasoning=False)

    assert "--served-model-name" in args
    assert "qwen36-fp8" in args
    assert "--reasoning-parser" not in args


def test_build_common_args_with_reasoning():
    args = build_common_args("qwen36-nvfp4", reasoning=True)

    assert "--reasoning-parser" in args
    idx = args.index("--reasoning-parser")
    assert args[idx + 1] == "qwen3"


def test_build_start_command_adds_variant_args_and_envs(monkeypatch):
    monkeypatch.setenv("VLLM_MARLIN_USE_ATOMIC_ADD", "0")
    monkeypatch.setenv("VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "0")

    command = build_start_command(
        variant="nvfp4",
        image="vllm-image",
        model="/model",
        container_name="vllm-nvfp4",
        common_args=["--host", "0.0.0.0", "--quantization", "modelopt"],
        host_cache_dir="/tmp/cache",
        restart_policy="unless-stopped",
        moe_backend="flashinfer_b12x",
        linear_backend=None,
        hf_token=None,
    )

    assert "--restart" in command
    assert "unless-stopped" in command
    assert "--quantization" in command
    assert "--moe-backend" in command and "flashinfer_b12x" in command
    assert "-v" in command and "/tmp/cache:/root/.cache/vllm" in command


def test_build_start_command_reuses_host_vllm_cache_dir():
    command = build_start_command(
        variant="nvfp4",
        image="vllm-image",
        model="/model",
        container_name="vllm-nvfp4",
        common_args=["--host", "0.0.0.0"],
        host_cache_dir="/tmp/custom-vllm-cache",
        restart_policy=None,
        moe_backend=None,
        linear_backend=None,
        hf_token=None,
    )

    assert "/tmp/custom-vllm-cache:/root/.cache/vllm" in command
    assert "-e" in command and "TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/torchinductor" in command

    fp8_command = build_start_command(
        variant="fp8",
        image="vllm-image",
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        container_name="vllm-fp8",
        common_args=["--host", "0.0.0.0"],
        host_cache_dir="/tmp/custom-vllm-cache",
        restart_policy=None,
        moe_backend=None,
        linear_backend=None,
        hf_token="x",
    )

    assert "/tmp/custom-vllm-cache:/root/.cache/vllm" in fp8_command
    assert "/root/.cache/huggingface" in " ".join(fp8_command)


def test_run_warmup_uses_sender_callable(tmp_path):
    calls = []

    def fake_sender(payload, max_time=120):
        calls.append(payload["prompt"])
        return 200, "{\"id\": \"ok\"}", None

    run_warmup("model", 2, "vllm-nvfp4", sender=fake_sender, output_dir=str(tmp_path))

    assert len(calls) == 2
    for i in (1, 2):
        p = tmp_path / f"vllm_warmup_vllm-nvfp4_{i}.json"
        assert p.exists()
        assert p.read_text() == '{"id": "ok"}'


def test_smoke_check_writes_file_when_ok(tmp_path):
    def fake_sender(payload, max_time=120):
        return 200, "{\"id\": \"ok\"}", None

    smoke_check("model", "vllm-nvfp4", sender=fake_sender, output_dir=str(tmp_path))
    p = tmp_path / "vllm_smoke_vllm-nvfp4.json"
    assert p.exists()
    assert p.read_text() == '{"id": "ok"}'


def test_wait_for_health_success():
    class FakeTailer:
        def __init__(self, name: str):
            self.name = name

        def start(self):
            pass

        def stop(self):
            pass

    health_seq = [False, True]

    def running(_name: str) -> bool:
        return True

    def healthy() -> bool:
        return health_seq.pop(0)

    ok = wait_for_health(
        "name",
        timeout_seconds=3,
        is_container_running=running,
        is_health_ok=healthy,
        tailer_factory=FakeTailer,
    )

    assert ok is True


def test_wait_for_health_container_exits():
    class FakeTailer:
        def __init__(self, name: str):
            self.name = name

        def start(self):
            pass

        def stop(self):
            pass

    ok = wait_for_health(
        "name",
        timeout_seconds=3,
        is_container_running=lambda name: False,
        is_health_ok=lambda: False,
        tailer_factory=FakeTailer,
    )

    assert ok is False


def test_run_detach_mode_keeps_container_running(monkeypatch, tmp_path):
    calls = {
        "remove": 0,
        "stream": 0,
    }

    def fake_start_server(**_kwargs: object) -> str:
        return "container-id"

    def fake_wait_for_health(name: str, _timeout_seconds: int, **_kwargs: object) -> bool:
        assert name == "vllm-nvfp4"
        return True

    def fake_remove_container(_name: str) -> None:
        calls["remove"] += 1

    def fake_stream_logs(_name: str, **_kwargs: object) -> int:
        calls["stream"] += 1
        raise AssertionError("stream_logs_forever should not run in detach mode")

    monkeypatch.setenv("VLLM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(orchestrator, "start_server", fake_start_server)
    monkeypatch.setattr(orchestrator, "wait_for_health", fake_wait_for_health)
    monkeypatch.setattr(orchestrator, "run_warmup", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "smoke_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "stream_logs_forever", fake_stream_logs)
    monkeypatch.setattr(orchestrator, "remove_container_if_exists", fake_remove_container)

    args = cli.LaunchArgs(
        variant="nvfp4",
        reasoning=False,
        no_warmup=False,
        no_smoke_check=False,
        enable_prefix_caching=False,
        detach=True,
        moe_backend=None,
        linear_backend=None,
        restart_policy=None,
    )

    code = orchestrator.run(args)

    assert code == 0
    assert calls["stream"] == 0
    assert calls["remove"] == 1


def test_run_stream_mode_stops_container_on_exit(monkeypatch, tmp_path):
    calls = {
        "remove": 0,
        "stream": 0,
    }

    def fake_start_server(**_kwargs: object) -> str:
        return "container-id"

    def fake_wait_for_health(name: str, _timeout_seconds: int, **_kwargs: object) -> bool:
        assert name == "vllm-fp8"
        return True

    def fake_remove_container(_name: str) -> None:
        calls["remove"] += 1

    def fake_stream_logs(_name: str, **_kwargs: object) -> int:
        calls["stream"] += 1
        return 0

    monkeypatch.setenv("VLLM_CACHE_DIR", str(tmp_path / "cache2"))
    monkeypatch.setattr(orchestrator, "start_server", fake_start_server)
    monkeypatch.setattr(orchestrator, "wait_for_health", fake_wait_for_health)
    monkeypatch.setattr(orchestrator, "run_warmup", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "smoke_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "stream_logs_forever", fake_stream_logs)
    monkeypatch.setattr(orchestrator, "remove_container_if_exists", fake_remove_container)

    args = cli.LaunchArgs(
        variant="fp8",
        reasoning=False,
        no_warmup=False,
        no_smoke_check=False,
        enable_prefix_caching=False,
        detach=False,
        moe_backend=None,
        linear_backend=None,
        restart_policy=None,
    )

    code = orchestrator.run(args)

    assert code == 0
    assert calls["stream"] == 1
    assert calls["remove"] == 2
