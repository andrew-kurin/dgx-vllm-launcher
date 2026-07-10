from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


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
        env_token = self._env.get("HF_TOKEN") or self._env.get("HUGGING_FACE_HUB_TOKEN")
        if env_token:
            return env_token.strip() or None

        configured_home = self._env.get("HF_HOME")
        candidate_roots = [
            Path(configured_home).expanduser() if configured_home else None,
            self._home_dir / ".cache" / "huggingface",
            self._home_dir / ".huggingface",
        ]
        for root in candidate_roots:
            if root is None:
                continue
            token_path = root / "token"
            try:
                token = token_path.read_text(encoding="utf-8").strip()
            except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
                continue
            if token:
                return token

        return None
