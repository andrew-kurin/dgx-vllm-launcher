from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .cli import LaunchArgs
from .config import (
    CONTAINER_PORT,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_HOST_PORT,
    DEFAULT_PRELOADED_MODELS_DIR,
    DEFAULT_READY_TIMEOUT,
    DEFAULT_VLLM_CACHE_DIR,
    Variant,
    resolve_variant_profile,
)
from .vllm_args import build_vllm_args


class ConfigurationError(ValueError):
    """Raised when launcher inputs cannot produce a valid launch plan."""


@dataclass(frozen=True)
class Mount:
    host_path: Path
    container_path: str
    read_only: bool = False


@dataclass(frozen=True)
class LaunchPlan:
    """Complete, validated, secret-free description of one launch."""

    variant: Variant
    model: str
    configured_model: str
    image: str
    served_model_name: str
    startup_message: str
    container_name: str
    host_port: int
    container_port: int
    ready_timeout_seconds: int
    warmup_requests: int
    run_smoke_check: bool
    detach: bool
    restart_policy: str | None
    requires_hf_token: bool
    inject_hf_token: bool
    uses_preloaded_model: bool
    preloaded_model_path: Path | None
    preloaded_models_root: Path
    artifact_dir: Path
    mounts: tuple[Mount, ...]
    container_env: tuple[tuple[str, str], ...]
    vllm_args: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.host_port}"


_RESTART_POLICY = re.compile(r"(?:no|always|unless-stopped|on-failure(?::\d+)?)")
_TUNED_CONFIG_CONTAINER_PATH = "/vllm-tuned-configs"


def _env_value(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default).strip()
    if not value:
        raise ConfigurationError(f"{name} must not be empty")
    return value


def _env_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw = env.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer; got {raw!r}") from exc

    if value < minimum or (maximum is not None and value > maximum):
        expected = (
            f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        )
        raise ConfigurationError(f"{name} must be {expected}; got {value}")
    return value


def _validate_restart_policy(policy: str | None) -> str | None:
    if policy is None:
        return None
    policy = policy.strip()
    if not _RESTART_POLICY.fullmatch(policy):
        raise ConfigurationError(
            "restart policy must be one of no, always, unless-stopped, "
            "on-failure, or on-failure:<retries>"
        )
    return policy


def _optional_backend(value: str | None, option: str) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ConfigurationError(f"{option} must not be empty")
    return value


def _resolved_path(raw_path: str) -> Path:
    return Path(raw_path).expanduser().resolve()


def _preloaded_candidate(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def resolve_launch_plan(
    args: LaunchArgs,
    env: Mapping[str, str] | None = None,
) -> LaunchPlan:
    """Resolve CLI inputs and environment into one validated launch plan."""

    env = os.environ if env is None else env
    if args.variant is None:
        raise ConfigurationError("variant is required for a launch")
    profile = resolve_variant_profile(args.variant)

    image = _env_value(env, profile.image_env_var, profile.default_image)
    ready_timeout = _env_int(
        env,
        "VLLM_READY_TIMEOUT",
        DEFAULT_READY_TIMEOUT,
        minimum=1,
    )
    host_port = _env_int(
        env,
        "VLLM_HOST_PORT",
        DEFAULT_HOST_PORT,
        minimum=1,
        maximum=65535,
    )
    warmup_requests = (
        0 if args.no_warmup else _env_int(env, "VLLM_WARMUP_REQUESTS", 2, minimum=0)
    )
    restart_policy = _validate_restart_policy(args.restart_policy)
    moe_backend = _optional_backend(args.moe_backend, "--moe-backend")
    if moe_backend is None:
        moe_backend = profile.default_moe_backend
    linear_backend = _optional_backend(args.linear_backend, "--linear-backend")
    if linear_backend is None:
        linear_backend = profile.default_linear_backend

    runtime_defaults = profile.runtime_defaults
    if not 0 < runtime_defaults.gpu_memory_utilization <= 1:
        raise ConfigurationError(
            "profile gpu_memory_utilization must be greater than 0 and at most 1"
        )
    if runtime_defaults.max_model_len <= 0:
        raise ConfigurationError("profile max_model_len must be positive")
    if runtime_defaults.max_num_seqs <= 0:
        raise ConfigurationError("profile max_num_seqs must be positive")
    if runtime_defaults.max_num_batched_tokens <= 0:
        raise ConfigurationError("profile max_num_batched_tokens must be positive")
    if args.reasoning and (
        not runtime_defaults.reasoning_parser or not runtime_defaults.tool_call_parser
    ):
        raise ConfigurationError(
            f"{args.variant} does not have reasoning/tool parsers configured"
        )

    cache_dir = _resolved_path(
        _env_value(env, "VLLM_CACHE_DIR", DEFAULT_VLLM_CACHE_DIR)
    )
    artifact_dir = _resolved_path(
        _env_value(env, "VLLM_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)
    )
    preloaded_root_raw = (
        args.preloaded_models_dir
        if args.preloaded_models_dir is not None
        else _env_value(
            env,
            "VLLM_PRELOADED_MODELS_DIR",
            DEFAULT_PRELOADED_MODELS_DIR,
        )
    )
    if not preloaded_root_raw.strip():
        raise ConfigurationError("preloaded models directory must not be empty")
    preloaded_root = _resolved_path(preloaded_root_raw)

    mounts = [Mount(cache_dir, "/root/.cache/vllm")]
    tuned_config_dir: Path | None = None
    if runtime_defaults.tuned_config_subdir is not None:
        package_root = Path(__file__).resolve().parent
        tuned_config_dir = (
            package_root / runtime_defaults.tuned_config_subdir
        ).resolve()
        if not tuned_config_dir.is_relative_to(package_root):
            raise ConfigurationError("profile tuned config directory escapes package")
        if not tuned_config_dir.is_dir():
            raise ConfigurationError(
                f"profile tuned config directory does not exist: {tuned_config_dir}"
            )
        mounts.append(
            Mount(
                tuned_config_dir,
                _TUNED_CONFIG_CONTAINER_PATH,
                read_only=True,
            )
        )
    warnings: list[str] = []
    model = profile.model
    uses_preloaded_model = False
    preloaded_model_path: Path | None = None
    preloaded = profile.source.preloaded
    if args.use_preloaded_models:
        if preloaded is None:
            warnings.append(
                f"Preloaded models requested, but {args.variant} has no preloaded "
                f"checkpoint configured; using {profile.model}."
            )
        else:
            candidate = _preloaded_candidate(
                preloaded_root,
                preloaded.relative_path,
            )
            if candidate.is_dir():
                uses_preloaded_model = True
                preloaded_model_path = candidate
                model = "/model"
                mounts.append(Mount(candidate, "/model", read_only=True))
            else:
                warnings.append(
                    f"Preloaded model not found at {candidate}; using {profile.model}."
                )

    token_policy = profile.source.token_policy
    requires_hf_token = token_policy == "required" and not uses_preloaded_model
    inject_hf_token = (
        token_policy in {"required", "optional"} and not uses_preloaded_model
    )
    if not uses_preloaded_model:
        hf_cache_default = env.get("HF_HOME", DEFAULT_HF_CACHE_DIR)
        hf_cache_dir = _resolved_path(
            _env_value(env, "VLLM_HF_CACHE_DIR", hf_cache_default)
        )
        mounts.append(Mount(hf_cache_dir, "/root/.cache/huggingface"))

    safetensors_strategy = _env_value(
        env,
        "VLLM_SAFETENSORS_LOAD_STRATEGY",
        "lazy",
    )
    container_env: list[tuple[str, str]] = [
        (
            "VLLM_MARLIN_USE_ATOMIC_ADD",
            _env_value(env, "VLLM_MARLIN_USE_ATOMIC_ADD", "1"),
        ),
        (
            "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE",
            _env_value(env, "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", "1"),
        ),
        ("TORCHINDUCTOR_CACHE_DIR", "/root/.cache/vllm/torchinductor"),
    ]
    if tuned_config_dir is not None:
        container_env.append(
            ("VLLM_TUNED_CONFIG_FOLDER", _TUNED_CONFIG_CONTAINER_PATH)
        )
    container_env.extend(runtime_defaults.container_env)

    vllm_args = build_vllm_args(
        profile.served_model_name,
        reasoning=args.reasoning,
        runtime_defaults=runtime_defaults,
        container_port=CONTAINER_PORT,
        safetensors_load_strategy=safetensors_strategy,
        quantization=profile.quantization,
        moe_backend=moe_backend,
        linear_backend=linear_backend,
    )
    startup_message = profile.startup_message
    if preloaded_model_path is not None:
        startup_message = (
            f"Serving {args.variant} from preloaded checkpoint {preloaded_model_path}."
        )

    return LaunchPlan(
        variant=args.variant,
        model=model,
        configured_model=profile.model,
        image=image,
        served_model_name=profile.served_model_name,
        startup_message=startup_message,
        container_name=f"vllm-{args.variant}",
        host_port=host_port,
        container_port=CONTAINER_PORT,
        ready_timeout_seconds=ready_timeout,
        warmup_requests=warmup_requests,
        run_smoke_check=not args.no_smoke_check,
        detach=args.detach,
        restart_policy=restart_policy,
        requires_hf_token=requires_hf_token,
        inject_hf_token=inject_hf_token,
        uses_preloaded_model=uses_preloaded_model,
        preloaded_model_path=preloaded_model_path,
        preloaded_models_root=preloaded_root,
        artifact_dir=artifact_dir,
        mounts=tuple(mounts),
        container_env=tuple(container_env),
        vllm_args=vllm_args,
        warnings=tuple(warnings),
    )
