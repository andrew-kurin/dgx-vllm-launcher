from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path

from .config import (
    CONTAINER_PORT,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BIND_ADDRESS,
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_HOST_PORT,
    DEFAULT_PRELOADED_MODELS_DIR,
    DEFAULT_READY_TIMEOUT,
    DEFAULT_VLLM_CACHE_DIR,
    Variant,
    VariantProfile,
    VariantRuntimeDefaults,
    resolve_variant_profile,
)
from .vllm_args import build_vllm_args


class ConfigurationError(ValueError):
    """Raised when launcher inputs cannot produce a valid launch plan."""


@dataclass(frozen=True)
class LaunchArgs:
    """User-selected launch settings, independent of their CLI representation."""

    variant: Variant | None
    reasoning: bool = False
    no_warmup: bool = False
    no_smoke_check: bool = False
    detach: bool = False
    moe_backend: str | None = None
    linear_backend: str | None = None
    restart_policy: str | None = None
    use_preloaded_models: bool = False
    preloaded_models_dir: str | None = None
    show_defaults: bool = False


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
    bind_address: str
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
    startup_python_packages: tuple[str, ...]
    vllm_args: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        address = self.bind_address
        if address == "0.0.0.0":
            address = "127.0.0.1"
        elif address == "::":
            address = "::1"
        host = f"[{address}]" if ":" in address else address
        return f"http://{host}:{self.host_port}"

    @property
    def docker_bind_address(self) -> str:
        """Format the validated address for Docker's published-port syntax."""

        if ":" in self.bind_address:
            return f"[{self.bind_address}]"
        return self.bind_address


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


def _env_bit(env: Mapping[str, str], name: str, default: int) -> str:
    raw = env.get(name, str(default)).strip()
    if raw not in {"0", "1"}:
        raise ConfigurationError(f"{name} must be 0 or 1; got {raw!r}")
    return raw


def _env_ip_address(env: Mapping[str, str], name: str, default: str) -> str:
    raw = _env_value(env, name, default)
    if "%" in raw:
        raise ConfigurationError(
            f"{name} must not include an IPv6 zone identifier; got {raw!r}"
        )
    try:
        return str(ip_address(raw))
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be an IPv4 or IPv6 address; got {raw!r}"
        ) from exc


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


def _resolve_allow_missing(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        # Cache and mount directories may be created after the plan is resolved.
        return path.resolve()


def _resolved_path(raw_path: str, *, setting: str) -> Path:
    try:
        return _resolve_allow_missing(Path(raw_path).expanduser())
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError(
            f"{setting} path could not be resolved: {raw_path!r}: {exc}"
        ) from exc


def _preloaded_candidate(
    root: Path,
    relative_path: str,
    *,
    root_setting: str,
) -> Path:
    candidate = Path(relative_path)
    try:
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return _resolve_allow_missing(candidate)
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError(
            f"preloaded model candidate under {root_setting} could not be resolved: "
            f"{candidate}: {exc}"
        ) from exc


def _resolve_image(env: Mapping[str, str], profile: VariantProfile) -> str:
    if profile.image_env_var in env:
        return _env_value(env, profile.image_env_var, profile.default_image)

    # TODO(remove backward-compat image env aliases): Remove legacy Qwen image
    # lookups in v0.2.0 after users migrate to the derived canonical names.
    if (
        profile.legacy_image_env_var is not None
        and profile.legacy_image_env_var in env
    ):
        return _env_value(
            env,
            profile.legacy_image_env_var,
            profile.default_image,
        )
    return profile.default_image


def _validate_reasoning_profile(
    profile: VariantProfile,
) -> None:
    runtime_defaults = profile.runtime_defaults
    if not runtime_defaults.reasoning_parser or not runtime_defaults.tool_call_parser:
        raise ConfigurationError(
            f"{profile.variant} does not have reasoning/tool parsers configured"
        )


@dataclass(frozen=True)
class _ResolvedModel:
    model: str
    uses_preloaded_model: bool
    preloaded_model_path: Path | None
    mounts: tuple[Mount, ...]
    warnings: tuple[str, ...]


def _resolve_model(
    args: LaunchArgs,
    profile: VariantProfile,
    preloaded_root: Path,
    preloaded_root_setting: str,
) -> _ResolvedModel:
    mounts: list[Mount] = []
    warnings: list[str] = []
    model = profile.model
    preloaded_model_path: Path | None = None
    preloaded = profile.source.preloaded
    if args.use_preloaded_models:
        if preloaded is None:
            warnings.append(
                f"Preloaded models requested, but {profile.variant} has no preloaded "
                f"checkpoint configured; using {profile.model}."
            )
        else:
            candidate = _preloaded_candidate(
                preloaded_root,
                preloaded.relative_path,
                root_setting=preloaded_root_setting,
            )
            if candidate.is_dir():
                preloaded_model_path = candidate
                model = "/model"
                mounts.append(Mount(candidate, "/model", read_only=True))
            else:
                warnings.append(
                    f"Preloaded model not found at {candidate}; using {profile.model}."
                )

    return _ResolvedModel(
        model=model,
        uses_preloaded_model=preloaded_model_path is not None,
        preloaded_model_path=preloaded_model_path,
        mounts=tuple(mounts),
        warnings=tuple(warnings),
    )


def _resolve_hf_cache_root(env: Mapping[str, str]) -> Path:
    if "VLLM_HF_CACHE_DIR" in env:
        raw_path = _env_value(env, "VLLM_HF_CACHE_DIR", DEFAULT_HF_CACHE_DIR)
        setting = "VLLM_HF_CACHE_DIR"
    else:
        configured_hf_home = env.get("HF_HOME", "").strip()
        raw_path = configured_hf_home or DEFAULT_HF_CACHE_DIR
        setting = "HF_HOME" if configured_hf_home else "VLLM_HF_CACHE_DIR/HF_HOME"
    return _resolved_path(raw_path, setting=setting)


def _build_container_env(
    env: Mapping[str, str],
    runtime_defaults: VariantRuntimeDefaults,
    *,
    tuned_config_dir: Path | None,
) -> tuple[tuple[str, str], ...]:
    container_env: list[tuple[str, str]] = [
        (
            "VLLM_MARLIN_USE_ATOMIC_ADD",
            _env_bit(env, "VLLM_MARLIN_USE_ATOMIC_ADD", 1),
        ),
        (
            "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE",
            _env_bit(env, "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE", 1),
        ),
        ("TORCHINDUCTOR_CACHE_DIR", "/root/.cache/vllm/torchinductor"),
    ]
    if tuned_config_dir is not None:
        container_env.append(
            ("VLLM_TUNED_CONFIG_FOLDER", _TUNED_CONFIG_CONTAINER_PATH)
        )
    container_env.extend(runtime_defaults.container_env)
    return tuple(container_env)


def resolve_launch_plan(
    args: LaunchArgs,
    env: Mapping[str, str] | None = None,
) -> LaunchPlan:
    """Resolve CLI inputs and environment into one validated launch plan."""

    env = os.environ if env is None else env
    if args.variant is None:
        raise ConfigurationError("variant is required for a launch")
    profile = resolve_variant_profile(args.variant)
    if args.reasoning:
        _validate_reasoning_profile(profile)

    image = _resolve_image(env, profile)
    bind_address = _env_ip_address(
        env,
        "VLLM_BIND_ADDRESS",
        DEFAULT_BIND_ADDRESS,
    )
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

    cache_dir = _resolved_path(
        _env_value(env, "VLLM_CACHE_DIR", DEFAULT_VLLM_CACHE_DIR),
        setting="VLLM_CACHE_DIR",
    )
    artifact_dir = _resolved_path(
        _env_value(env, "VLLM_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR),
        setting="VLLM_ARTIFACT_DIR",
    )
    if args.preloaded_models_dir is not None:
        preloaded_root_raw = args.preloaded_models_dir
        preloaded_root_setting = "--preloaded-models-dir"
    else:
        preloaded_root_raw = _env_value(
            env,
            "VLLM_PRELOADED_MODELS_DIR",
            DEFAULT_PRELOADED_MODELS_DIR,
        )
        preloaded_root_setting = "VLLM_PRELOADED_MODELS_DIR"
    if not preloaded_root_raw.strip():
        raise ConfigurationError(f"{preloaded_root_setting} must not be empty")
    preloaded_root = _resolved_path(
        preloaded_root_raw,
        setting=preloaded_root_setting,
    )

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
    resolved_model = _resolve_model(
        args,
        profile,
        preloaded_root,
        preloaded_root_setting,
    )
    mounts.extend(resolved_model.mounts)

    token_policy = profile.source.token_policy
    requires_hf_token = (
        token_policy == "required" and not resolved_model.uses_preloaded_model
    )
    inject_hf_token = (
        token_policy in {"required", "optional"}
        and not resolved_model.uses_preloaded_model
    )
    if not resolved_model.uses_preloaded_model:
        hf_cache_root = _resolve_hf_cache_root(env)
        mounts.extend(
            (
                Mount(hf_cache_root / "hub", "/root/.cache/huggingface/hub"),
                Mount(hf_cache_root / "xet", "/root/.cache/huggingface/xet"),
            )
        )

    safetensors_strategy = _env_value(
        env,
        "VLLM_SAFETENSORS_LOAD_STRATEGY",
        "lazy",
    )
    container_env = _build_container_env(
        env,
        runtime_defaults,
        tuned_config_dir=tuned_config_dir,
    )

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
    if resolved_model.preloaded_model_path is not None:
        startup_message = (
            f"Serving {args.variant} from preloaded checkpoint "
            f"{resolved_model.preloaded_model_path}."
        )

    return LaunchPlan(
        variant=args.variant,
        model=resolved_model.model,
        configured_model=profile.model,
        image=image,
        served_model_name=profile.served_model_name,
        startup_message=startup_message,
        container_name=f"vllm-{args.variant}",
        bind_address=bind_address,
        host_port=host_port,
        container_port=CONTAINER_PORT,
        ready_timeout_seconds=ready_timeout,
        warmup_requests=warmup_requests,
        run_smoke_check=not args.no_smoke_check,
        detach=args.detach,
        restart_policy=restart_policy,
        requires_hf_token=requires_hf_token,
        inject_hf_token=inject_hf_token,
        uses_preloaded_model=resolved_model.uses_preloaded_model,
        preloaded_model_path=resolved_model.preloaded_model_path,
        preloaded_models_root=preloaded_root,
        artifact_dir=artifact_dir,
        mounts=tuple(mounts),
        container_env=container_env,
        startup_python_packages=runtime_defaults.startup_python_packages,
        vllm_args=vllm_args,
        warnings=resolved_model.warnings,
    )
