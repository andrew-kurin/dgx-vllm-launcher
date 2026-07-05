from __future__ import annotations

from dgx_vllm_launcher import cli
import dgx_vllm_launcher.orchestrator as orchestrator
from dgx_vllm_launcher.orchestrator import (
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
        variant="qwen36-nvfp4",
        image="vllm-image",
        model="/model",
        container_name="vllm-qwen36-nvfp4",
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
        variant="qwen36-nvfp4",
        image="vllm-image",
        model="/model",
        container_name="vllm-qwen36-nvfp4",
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
        variant="qwen36-fp8",
        image="vllm-image",
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        container_name="vllm-qwen36-fp8",
        common_args=["--host", "0.0.0.0"],
        host_cache_dir="/tmp/custom-vllm-cache",
        restart_policy=None,
        moe_backend=None,
        linear_backend=None,
        hf_token="x",
    )

    assert "/tmp/custom-vllm-cache:/root/.cache/vllm" in fp8_command
    assert "/root/.cache/huggingface" in " ".join(fp8_command)


def test_build_start_command_gemma4_does_not_mount_local_qwen_model(monkeypatch):
    command = build_start_command(
        variant="gemma4-nvfp4",
        image="vllm-image",
        model="nvidia/Gemma-4-26B-A4B-NVFP4",
        container_name="vllm-gemma4-nvfp4",
        common_args=["--host", "0.0.0.0", "--quantization", "modelopt"],
        host_cache_dir="/tmp/cache",
        restart_policy=None,
        moe_backend=None,
        linear_backend=None,
        hf_token=None,
    )

    assert "/tmp/cache:/root/.cache/vllm" in " ".join(command)
    assert "Qwen3.6-35B-A3B-NVFP4" not in " ".join(command)
    assert "/root/.cache/huggingface" not in " ".join(command)


def test_build_start_command_ornith4_does_not_mount_local_model(monkeypatch):
    command = build_start_command(
        variant="ornith-nvfp4",
        image="vllm-image",
        model="sakamakismile/Ornith-1.0-35B-NVFP4",
        container_name="vllm-ornith-nvfp4",
        common_args=["--host", "0.0.0.0", "--quantization", "modelopt"],
        host_cache_dir="/tmp/cache",
        restart_policy=None,
        moe_backend=None,
        linear_backend=None,
        hf_token=None,
    )

    assert "/tmp/cache:/root/.cache/vllm" in " ".join(command)
    assert "Qwen3.6-35B-A3B-NVFP4" not in " ".join(command)
    assert "--quantization" in command
    assert "modelopt" in command
    assert "/root/.cache/huggingface" not in " ".join(command)


def test_run_warmup_uses_sender_callable(tmp_path):
    calls = []

    def fake_sender(payload, max_time=120):
        calls.append(payload["prompt"])
        return 200, "{\"id\": \"ok\"}", None

    run_warmup("model", 2, "vllm-qwen36-nvfp4", sender=fake_sender, output_dir=str(tmp_path))

    assert len(calls) == 2
    for i in (1, 2):
        p = tmp_path / f"vllm_warmup_vllm-qwen36-nvfp4_{i}.json"
        assert p.exists()
        assert p.read_text() == '{"id": "ok"}'


def test_smoke_check_writes_file_when_ok(tmp_path):
    def fake_sender(payload, max_time=120):
        return 200, "{\"id\": \"ok\"}", None

    smoke_check("model", "vllm-qwen36-nvfp4", sender=fake_sender, output_dir=str(tmp_path))
    p = tmp_path / "vllm_smoke_vllm-qwen36-nvfp4.json"
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
        assert name == "vllm-qwen36-nvfp4"
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
        variant="qwen36-nvfp4",
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
        assert name == "vllm-qwen36-fp8"
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
        variant="qwen36-fp8",
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


def test_run_show_defaults_without_starting_container(monkeypatch, tmp_path):
    calls = {"printed": 0}

    def fail_start_server(**_kwargs: object) -> str:
        raise AssertionError("start_server should not be called when --show-defaults is set")

    monkeypatch.setattr(orchestrator, "print_default_launch_profiles", lambda: calls.__setitem__("printed", calls["printed"] + 1))
    monkeypatch.setattr(orchestrator, "start_server", fail_start_server)
    monkeypatch.setenv("VLLM_CACHE_DIR", str(tmp_path / "cache-defaults"))

    args = cli.LaunchArgs(
        variant="qwen36-fp8",
        reasoning=False,
        no_warmup=False,
        no_smoke_check=False,
        enable_prefix_caching=False,
        detach=False,
        moe_backend=None,
        linear_backend=None,
        restart_policy=None,
        show_defaults=True,
    )

    code = orchestrator.run(args)

    assert code == 0
    assert calls["printed"] == 1


def test_run_qwen36_nvfp4_defaults_moe_backend(monkeypatch, tmp_path):
    captured_kwargs = {}

    def fake_start_server(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "container-id"

    def fake_wait_for_health(name: str, _timeout_seconds: int, **_kwargs: object) -> bool:
        assert name == "vllm-qwen36-nvfp4"
        return True

    monkeypatch.setenv("VLLM_CACHE_DIR", str(tmp_path / "cache3"))
    monkeypatch.setattr(orchestrator, "start_server", fake_start_server)
    monkeypatch.setattr(orchestrator, "wait_for_health", fake_wait_for_health)
    monkeypatch.setattr(orchestrator, "run_warmup", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "smoke_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "stream_logs_forever", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(orchestrator, "remove_container_if_exists", lambda _name: None)

    args = cli.LaunchArgs(
        variant="qwen36-nvfp4",
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
    assert captured_kwargs["moe_backend"] == "flashinfer_b12x"


def test_run_gemma4_no_default_moe_backend(monkeypatch, tmp_path):
    captured_kwargs = {}

    def fake_start_server(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "container-id"

    def fake_wait_for_health(name: str, _timeout_seconds: int, **_kwargs: object) -> bool:
        assert name == "vllm-gemma4-nvfp4"
        return True

    monkeypatch.setenv("VLLM_CACHE_DIR", str(tmp_path / "cache-gemma4"))
    monkeypatch.setattr(orchestrator, "start_server", fake_start_server)
    monkeypatch.setattr(orchestrator, "wait_for_health", fake_wait_for_health)
    monkeypatch.setattr(orchestrator, "run_warmup", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "smoke_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "stream_logs_forever", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(orchestrator, "remove_container_if_exists", lambda _name: None)

    args = cli.LaunchArgs(
        variant="gemma4-nvfp4",
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
    assert captured_kwargs["moe_backend"] is None
    assert captured_kwargs["model"] == "nvidia/Gemma-4-26B-A4B-NVFP4"


def test_run_qwen36_fp8_does_not_set_default_moe_backend(monkeypatch, tmp_path):
    captured_kwargs = {}

    def fake_start_server(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "container-id"

    def fake_wait_for_health(name: str, _timeout_seconds: int, **_kwargs: object) -> bool:
        assert name == "vllm-qwen36-fp8"
        return True

    monkeypatch.setenv("VLLM_CACHE_DIR", str(tmp_path / "cache4"))
    monkeypatch.setattr(orchestrator, "start_server", fake_start_server)
    monkeypatch.setattr(orchestrator, "wait_for_health", fake_wait_for_health)
    monkeypatch.setattr(orchestrator, "run_warmup", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "smoke_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "stream_logs_forever", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(orchestrator, "remove_container_if_exists", lambda _name: None)

    args = cli.LaunchArgs(
        variant="qwen36-fp8",
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
    assert captured_kwargs["moe_backend"] is None


def test_run_gemma4_allows_explicit_moe_backend(monkeypatch, tmp_path):
    captured_kwargs = {}

    def fake_start_server(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "container-id"

    def fake_wait_for_health(name: str, _timeout_seconds: int, **_kwargs: object) -> bool:
        assert name == "vllm-gemma4-nvfp4"
        return True

    monkeypatch.setenv("VLLM_CACHE_DIR", str(tmp_path / "cache-gemma4-explicit"))
    monkeypatch.setattr(orchestrator, "start_server", fake_start_server)
    monkeypatch.setattr(orchestrator, "wait_for_health", fake_wait_for_health)
    monkeypatch.setattr(orchestrator, "run_warmup", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "smoke_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "stream_logs_forever", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(orchestrator, "remove_container_if_exists", lambda _name: None)

    args = cli.LaunchArgs(
        variant="gemma4-nvfp4",
        reasoning=False,
        no_warmup=False,
        no_smoke_check=False,
        enable_prefix_caching=False,
        detach=False,
        moe_backend="flashinfer_b12x",
        linear_backend=None,
        restart_policy=None,
    )

    code = orchestrator.run(args)

    assert code == 0
    assert captured_kwargs["moe_backend"] == "flashinfer_b12x"
    assert captured_kwargs["model"] == "nvidia/Gemma-4-26B-A4B-NVFP4"


def test_run_ornith_no_default_moe_backend(monkeypatch, tmp_path):
    captured_kwargs = {}

    def fake_start_server(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        return "container-id"

    def fake_wait_for_health(name: str, _timeout_seconds: int, **_kwargs: object) -> bool:
        assert name == "vllm-ornith-nvfp4"
        return True

    monkeypatch.setenv("VLLM_CACHE_DIR", str(tmp_path / "cache-ornith"))
    monkeypatch.setattr(orchestrator, "start_server", fake_start_server)
    monkeypatch.setattr(orchestrator, "wait_for_health", fake_wait_for_health)
    monkeypatch.setattr(orchestrator, "run_warmup", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "smoke_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "stream_logs_forever", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(orchestrator, "remove_container_if_exists", lambda _name: None)

    args = cli.LaunchArgs(
        variant="ornith-nvfp4",
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
    assert captured_kwargs["moe_backend"] is None
    assert captured_kwargs["model"] == "sakamakismile/Ornith-1.0-35B-NVFP4"


def test_resolve_hf_token_prefers_env_over_file(monkeypatch, tmp_path):
    token_path = tmp_path / "token"
    token_path.write_text("file-token")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "env-token")

    token = orchestrator._resolve_hf_token()

    assert token == "env-token"


def test_resolve_hf_token_from_hf_home(monkeypatch, tmp_path):
    token_path = tmp_path / "token"
    token_path.write_text("home-token")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    token = orchestrator._resolve_hf_token()

    assert token == "home-token"
