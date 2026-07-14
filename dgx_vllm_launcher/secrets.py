from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from .plan import ConfigurationError


class HuggingFaceTokenProvider:
    """Resolve a Hugging Face token without putting it into a command string."""

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        *,
        home_dir: Path | None = None,
    ) -> None:
        self._env = os.environ if env is None else env
        self._home_dir = Path.home() if home_dir is None else home_dir

    def get_hf_token(self) -> str | None:
        # TODO(remove backward-compat token env alias): Remove
        # HUGGING_FACE_HUB_TOKEN in v0.2.0 after users migrate to HF_TOKEN.
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env_token = self._env.get(name, "").strip()
            if env_token:
                return env_token

        configured_home = self._env.get("HF_HOME", "").strip()
        try:
            configured_root = (
                Path(configured_home).expanduser() if configured_home else None
            )
        except (OSError, RuntimeError) as exc:
            raise ConfigurationError(
                f"HF_HOME path could not be expanded: {configured_home!r}: {exc}"
            ) from exc
        candidate_roots = [
            configured_root,
            self._home_dir / ".cache" / "huggingface",
        ]
        # TODO(remove backward-compat token path): Remove ~/.huggingface/token
        # lookup in v0.2.0 after users migrate to HF_HOME or
        # ~/.cache/huggingface/token.
        candidate_roots.append(self._home_dir / ".huggingface")
        for root in candidate_roots:
            if root is None:
                continue
            token_path = root / "token"
            try:
                token = token_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                continue
            if token:
                return token

        return None
