from __future__ import annotations

from pathlib import Path

from dgx_vllm_launcher.secrets import HuggingFaceTokenProvider


def test_token_provider_prefers_environment(tmp_path: Path):
    (tmp_path / "token").write_text("file-token", encoding="utf-8")
    provider = HuggingFaceTokenProvider(
        {"HF_HOME": str(tmp_path), "HF_TOKEN": "env-token"},
        home_dir=tmp_path,
    )

    assert provider.get_hf_token() == "env-token"


def test_token_provider_reads_configured_hf_home(tmp_path: Path):
    (tmp_path / "token").write_text("home-token\n", encoding="utf-8")
    provider = HuggingFaceTokenProvider(
        {"HF_HOME": str(tmp_path)},
        home_dir=tmp_path / "unused-home",
    )

    assert provider.get_hf_token() == "home-token"


def test_token_provider_returns_none_when_unconfigured(tmp_path: Path):
    assert HuggingFaceTokenProvider({}, home_dir=tmp_path).get_hf_token() is None
