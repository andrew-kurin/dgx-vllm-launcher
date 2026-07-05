from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Callable, Protocol

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .cli import LaunchArgs, parse_args
from .config import (
    VARIANTS,
    Variant,
    VariantConfig,
    VariantProfile,
    resolve_cache_dir,
    resolve_env_int,
    resolve_preloaded_models_root,
    resolve_variant_config,
    resolve_variant_profile,
)
from .docker_ops import (
    container_logs,
    container_running,
    inspect_container,
    remove_container,
    run_docker,
    stop_container,
    stream_container_logs,
)
from .http_ops import health_ok, request_completion
from .vllm_args import build_common_args


console = Console()


@dataclass(frozen=True)
class LaunchPlan:
    variant_config: VariantConfig
    profile: VariantProfile
    warmup_requests: int
    host_cache_dir: str
    common_args: list[str]


class _LogTailer(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


class LogTailer:
    def __init__(self, container_name: str):
        self.container_name = container_name
        self._proc: subprocess.Popen[str] | None = None
        self._running = False
        self._reader: Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._proc = stream_container_logs(self.container_name, capture_output=True)
        self._running = True

        # Keep reading in a lightweight loop until stopped or command exits.

        def _reader() -> None:
            assert self._proc and self._proc.stdout
            if self._proc.stdout is None:
                return
            for line in self._proc.stdout:
                if not self._running:
                    return
                line = line.rstrip("\n")
                if line:
                    console.print(f"[dim]{line}[/]")

        thread = Thread(target=_reader, daemon=True)
        thread.start()
        self._reader = thread

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=1)

        self._proc = None


def _resolve_hf_token() -> str | None:
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token.strip() or None

    hf_home = os.environ.get("HF_HOME")
    candidate_roots = [
        Path(hf_home) if hf_home else None,
        Path("~/.cache/huggingface").expanduser(),
        Path("~/.huggingface").expanduser(),
    ]
    for root in candidate_roots:
        if not root:
            continue
        token_path = root / "token"
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                return token

    return None


def _resolve_preloaded_model_path(local_model_path: str, preloaded_models_dir: str) -> str:
    expanded_path = os.path.expanduser(local_model_path)
    if os.path.isabs(expanded_path):
        return expanded_path

    return os.path.join(preloaded_models_dir, expanded_path)


def _preloaded_model_candidate(profile: VariantProfile, preloaded_models_dir: str) -> str | None:
    if not profile.mount_local_model or not profile.local_model_path:
        return None
    return _resolve_preloaded_model_path(profile.local_model_path, preloaded_models_dir)


def build_start_command(
    *,
    variant: Variant,
    image: str,
    model: str,
    container_name: str,
    common_args: list[str],
    host_cache_dir: str,
    restart_policy: str | None,
    moe_backend: str | None,
    linear_backend: str | None,
    hf_token: str | None,
    use_preloaded_models: bool = False,
    preloaded_models_dir: str | None = None,
) -> list[str]:
    vllm_marin = os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "1")
    vllm_inductor = os.environ.get("VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "1")
    vllm_nvfp4_gemm_backend = os.environ.get("VLLM_NVFP4_GEMM_BACKEND")
    profile = resolve_variant_profile(variant)

    cmd = [
        "docker",
        "run",
        "-d",
        "--gpus",
        "all",
        "-p",
        "8000:8000",
        "--name",
        container_name,
        "--ipc=host",
        "-e",
        f"VLLM_MARLIN_USE_ATOMIC_ADD={vllm_marin}",
        "-e",
        f"VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE={vllm_inductor}",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/torchinductor",
        "-v",
        f"{host_cache_dir}:/root/.cache/vllm",
    ]

    if vllm_nvfp4_gemm_backend:
        cmd.extend(["-e", f"VLLM_NVFP4_GEMM_BACKEND={vllm_nvfp4_gemm_backend}"])

    if restart_policy:
        cmd.extend(["--restart", restart_policy])

    local_model_path = None
    if use_preloaded_models:
        preloaded_root = preloaded_models_dir or resolve_preloaded_models_root()
        candidate_path = _preloaded_model_candidate(profile, preloaded_root)
        if candidate_path and os.path.isdir(candidate_path):
            local_model_path = candidate_path
            cmd.extend(["-v", f"{local_model_path}:/model"])
            model = "/model"

    needs_hf_token = profile.requires_hf_token or profile.inject_hf_token
    if needs_hf_token and not hf_token and not local_model_path:
        if profile.requires_hf_token:
            raise RuntimeError(f"HF token required for {variant}; set HF_TOKEN or run `huggingface-cli login` and retry")
    if hf_token and needs_hf_token and not local_model_path:
        hf_cache = os.path.expanduser("~/.cache/huggingface")
        cmd.extend(
            [
                "-e",
                f"HF_TOKEN={hf_token}",
                "-v",
                f"{hf_cache}:/root/.cache/huggingface",
            ]
        )

    if moe_backend:
        common_args = list(common_args)
        common_args.extend(["--moe-backend", moe_backend])
    if linear_backend:
        common_args = list(common_args)
        common_args.extend(["--linear-backend", linear_backend])

    cmd.extend([image, model])
    cmd.extend(common_args)
    return cmd


def start_server(
    *,
    variant: Variant,
    image: str,
    model: str,
    container_name: str,
    common_args: list[str],
    moe_backend: str | None,
    linear_backend: str | None,
    restart_policy: str | None,
    host_cache_dir: str,
    use_preloaded_models: bool = False,
    preloaded_models_dir: str | None = None,
) -> str:
    profile = resolve_variant_profile(variant)
    hf_token = _resolve_hf_token() if (profile.requires_hf_token or profile.inject_hf_token) else None

    cmd = build_start_command(
        variant=variant,
        image=image,
        model=model,
        container_name=container_name,
        common_args=common_args,
        host_cache_dir=host_cache_dir,
        restart_policy=restart_policy,
        moe_backend=moe_backend,
        linear_backend=linear_backend,
        hf_token=hf_token,
        use_preloaded_models=use_preloaded_models,
        preloaded_models_dir=preloaded_models_dir,
    )

    proc = run_docker(cmd, check=True, capture_output=True)

    container_id = (proc.stdout or "").strip()
    if not container_id:
        raise RuntimeError("docker run produced no container ID")
    return container_id


def wait_for_health(
    name: str,
    timeout_seconds: int,
    *,
    is_container_running: Callable[[str], bool] = container_running,
    is_health_ok: Callable[[], bool] = health_ok,
    tailer_factory: Callable[[str], _LogTailer] = LogTailer,
) -> bool:
    tailer = tailer_factory(name)
    tailer.start()
    try:
        with console.status("[bold cyan]Waiting for readiness on /health ...[/]", spinner="dots") as status:
            for attempt in range(1, timeout_seconds + 1):
                status.update(f"[bold cyan]Waiting for readiness on /health ({attempt}/{timeout_seconds})[/]")
                if not is_container_running(name):
                    console.print("[red]Container exited before becoming ready.[/red]")
                    return False

                if is_health_ok():
                    return True

                time.sleep(1)

        console.print("[yellow]Timed out waiting for health endpoint.[/yellow]")
        return False
    finally:
        tailer.stop()


def run_warmup(
    model_name: str,
    requests_count: int,
    container_name: str,
    *,
    sender=request_completion,
    output_dir: str = "/tmp",
) -> None:
    if requests_count <= 0:
        console.print("[yellow]Warmup skipped.[/yellow]")
        return

    console.print(f"\n[blue]Running {requests_count} warmup completion(s)...[/blue]")
    warmup_prompt = (
        "The purpose of this request is warmup only. Please ignore the content "
        "and return a short completion immediately without special formatting."
    )

    for i in range(1, requests_count + 1):
        payload = {
            "model": model_name,
            "prompt": warmup_prompt,
            "max_tokens": 2,
            "temperature": 0.0,
            "return_token_ids": True,
        }
        status, body, err = sender(payload, max_time=120)
        path = f"{output_dir}/vllm_warmup_{container_name}_{i}.json"
        if status == 200 and body is not None:
            Path(path).write_text(body)
            console.print(f"  [green]✓[/green] warmup {i}/{requests_count} complete")
        else:
            console.print(f"  [red]✗[/red] warmup {i}/{requests_count} failed")
            if err:
                console.print(f"    [dim]{err}[/dim]")


def smoke_check(
    model_name: str,
    container_name: str,
    *,
    sender=request_completion,
    output_dir: str = "/tmp",
) -> None:
    payload = {
        "model": model_name,
        "prompt": "Smoke test request.",
        "max_tokens": 4,
        "temperature": 0.0,
        "return_token_ids": True,
    }
    status, body, err = sender(payload, max_time=120)

    if status == 200 and body is not None:
        path = f"{output_dir}/vllm_smoke_{container_name}.json"
        Path(path).write_text(body)
        console.print(f"[green]Smoke check passed.[/green] Response saved to {path}")
        for line in body.splitlines()[:3]:
            console.print(f"[dim]{line}[/dim]")
        return

    console.print("[yellow]warning: smoke check request failed[/yellow]")
    if err:
        console.print(f"[dim]{err}[/]")


def _format_common_args(args: list[str]) -> Table:
    table = Table(
        title="Common args",
        box=box.ROUNDED,
        show_header=False,
        show_edge=False,
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Arg", style="cyan")
    table.add_column("Value", style="magenta")

    no_value_flags = {
        "--enable-prefix-caching",
        "--enable-flashinfer-autotune",
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--async-scheduling",
    }
    value_flags = {
        "--host",
        "--port",
        "--tensor-parallel-size",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--safetensors-load-strategy",
        "--generation-config",
        "--served-model-name",
        "--reasoning-parser",
        "--tool-call-parser",
        "--moe-backend",
        "--linear-backend",
        "--quantization",
        "--chat-template",
        "--kv-cache-dtype",
        "--limit-mm-per-prompt",
        "--mm-processor-kwargs",
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in no_value_flags:
            table.add_row(arg, "")
            i += 1
            continue

        if arg in value_flags and i + 1 < len(args):
            table.add_row(arg, args[i + 1])
            i += 2
            continue

        if i == len(args) - 1 or args[i + 1].startswith("--"):
            table.add_row(arg, "")
            i += 1
            continue

        table.add_row(arg, args[i + 1])
        i += 2

    return table


def print_summary(
    *,
    variant: str,
    model: str,
    image: str,
    served_name: str,
    common_args: list[str],
    restart_policy: str | None,
) -> None:
    table = Table(title="Launcher settings", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Variant", variant)
    table.add_row("Model", model)
    table.add_row("Image", image)
    table.add_row("Container", f"vllm-{variant}")
    table.add_row("Served model name", served_name)
    table.add_row("Restart policy", restart_policy or "(none)")
    console.print(Panel(table, title="vLLM serve", expand=False))
    if common_args:
        console.print(_format_common_args(common_args))


def print_default_launch_profiles() -> None:
    table = Table(title="Default launch configuration by variant", box=box.ROUNDED)
    table.add_column("Variant", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Default MoE backend", style="yellow")
    table.add_column("Default linear backend", style="yellow")
    table.add_column("Quantization", style="blue")

    for variant in VARIANTS:
        profile = resolve_variant_profile(variant)
        table.add_row(
            variant,
            profile.model,
            profile.default_moe_backend or "(none)",
            profile.default_linear_backend or "(none)",
            profile.quantization or "(none)",
        )

    console.print(Panel(table, title="Recommended defaults", expand=False))


def stream_logs_forever(name: str, *, tail: int | None = 20) -> int:
    proc = stream_container_logs(name, tail=tail)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        raise


def remove_container_if_exists(name: str) -> None:
    if inspect_container(name).returncode != 0:
        return
    console.print(f"[yellow]Cleaning up container {name}[/yellow]")
    stop_container(name)
    remove_container(name)


def _signal_to_keyboard_interrupt(_signum: int, _frame) -> None:
    raise KeyboardInterrupt


def _build_launch_config(args: LaunchArgs) -> LaunchPlan:
    variant = args.variant
    if variant is None:
        raise RuntimeError("variant is required unless --show-defaults is set")

    variant_config = resolve_variant_config(variant)
    profile = resolve_variant_profile(variant)
    warmup_requests = 0 if args.no_warmup else resolve_env_int("VLLM_WARMUP_REQUESTS", 2)
    host_cache_dir = resolve_cache_dir()
    common_args = build_common_args(
        variant_config.served_model_name,
        args.reasoning,
        profile.runtime_defaults,
    )
    if profile.quantization:
        common_args.extend(["--quantization", profile.quantization])

    return LaunchPlan(
        variant_config=variant_config,
        profile=profile,
        warmup_requests=warmup_requests,
        host_cache_dir=host_cache_dir,
        common_args=common_args,
    )


def run(args: LaunchArgs) -> int:
    signal.signal(signal.SIGINT, _signal_to_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _signal_to_keyboard_interrupt)

    if args.show_defaults:
        print_default_launch_profiles()
        return 0

    variant = args.variant
    if variant is None:
        raise RuntimeError("variant is required unless --show-defaults is set")

    plan = _build_launch_config(args)
    container_name = f"vllm-{variant}"
    timeout_seconds = plan.variant_config.ready_timeout_seconds
    Path(plan.host_cache_dir).mkdir(parents=True, exist_ok=True)

    print_summary(
        variant=variant,
        model=plan.variant_config.model,
        image=plan.variant_config.image,
        served_name=plan.variant_config.served_model_name,
        common_args=plan.common_args,
        restart_policy=args.restart_policy,
    )

    remove_container_if_exists(container_name)

    started = False
    cleanup_container = True
    try:
        console.print(f"[green]{plan.profile.startup_message}[/green]")

        resolved_moe_backend = args.moe_backend
        if resolved_moe_backend is None:
            resolved_moe_backend = plan.profile.default_moe_backend

        resolved_linear_backend = args.linear_backend
        if resolved_linear_backend is None:
            resolved_linear_backend = plan.profile.default_linear_backend

        preloaded_models_dir = resolve_preloaded_models_root(args.preloaded_models_dir)
        if args.use_preloaded_models:
            candidate_path = _preloaded_model_candidate(plan.profile, preloaded_models_dir)
            if candidate_path is None:
                console.print(f"[yellow]Preloaded models requested, but {variant} has no preloaded model path configured; using configured model ID.[/yellow]")
            elif not os.path.isdir(candidate_path):
                console.print(f"[yellow]Preloaded model not found at {candidate_path}; using configured model ID {plan.variant_config.model}.[/yellow]")

        container_id = start_server(
            variant=variant,
            image=plan.variant_config.image,
            model=plan.variant_config.model,
            container_name=container_name,
            common_args=plan.common_args,
            moe_backend=resolved_moe_backend,
            linear_backend=resolved_linear_backend,
            restart_policy=args.restart_policy,
            host_cache_dir=plan.host_cache_dir,
            use_preloaded_models=args.use_preloaded_models,
            preloaded_models_dir=preloaded_models_dir,
        )
        started = True

        console.print(f"[green]Started container {container_id} as {container_name}[/green]")
        console.print("[blue]Waiting for readiness on /health ...[/blue]")
        if not wait_for_health(container_name, timeout_seconds):
            logs = container_logs(container_name, tail=200)
            console.print(Panel(logs, title="startup logs", border_style="red", expand=False))
            return 1

        console.print("[green]Service is healthy. Running warmup + smoke readiness checks.[/green]")
        run_warmup(plan.variant_config.served_model_name, plan.warmup_requests, container_name)
        if not args.no_smoke_check:
            smoke_check(plan.variant_config.served_model_name, container_name)

        if args.detach:
            cleanup_container = False
            console.print("\n[green]Startup checks passed; container is now running in detached mode.[/green]")
            console.print(f"[blue]Service container:[/] vllm-{variant}")
            console.print(f"[blue]Tail logs with:[/] docker logs -f vllm-{variant}")
            console.print(f"[blue]Stop container with:[/] docker stop vllm-{variant}")
            if args.restart_policy:
                console.print(f"[blue]Restart policy:[/] {args.restart_policy} (Docker will attempt restart on failures/restarts).[/blue]")
            return 0

        console.print("\n[blue]Streaming logs. Press Ctrl-C to stop; container will be stopped.[/blue]")
        return stream_logs_forever(container_name)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Exiting...[/yellow]")
        return 130
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        return 1
    finally:
        if started and cleanup_container:
            remove_container_if_exists(container_name)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


def main_entrypoint() -> int:
    return main()
