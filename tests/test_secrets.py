from __future__ import annotations

from pathlib import Path

import pytest

from dgx_vllm_launcher.plan import ConfigurationError
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


def test_token_provider_reads_hugging_face_hub_token_alias(tmp_path: Path):
    provider = HuggingFaceTokenProvider(
        {"HUGGING_FACE_HUB_TOKEN": " alias-token\n"},
        home_dir=tmp_path,
    )

    assert provider.get_hf_token() == "alias-token"


def test_whitespace_hf_token_does_not_hide_alias(tmp_path: Path):
    provider = HuggingFaceTokenProvider(
        {
            "HF_TOKEN": "   ",
            "HUGGING_FACE_HUB_TOKEN": "alias-token",
        },
        home_dir=tmp_path,
    )

    assert provider.get_hf_token() == "alias-token"


def test_whitespace_environment_tokens_do_not_hide_default_cache_file(
    tmp_path: Path,
):
    cache_root = tmp_path / ".cache" / "huggingface"
    cache_root.mkdir(parents=True)
    (cache_root / "token").write_text("cache-token\n", encoding="utf-8")
    provider = HuggingFaceTokenProvider(
        {"HF_TOKEN": " ", "HUGGING_FACE_HUB_TOKEN": "\t"},
        home_dir=tmp_path,
    )

    assert provider.get_hf_token() == "cache-token"


def test_token_provider_reads_legacy_huggingface_file(tmp_path: Path):
    legacy_root = tmp_path / ".huggingface"
    legacy_root.mkdir()
    (legacy_root / "token").write_text("legacy-token", encoding="utf-8")

    provider = HuggingFaceTokenProvider({}, home_dir=tmp_path)

    assert provider.get_hf_token() == "legacy-token"


def test_invalid_utf8_token_file_falls_through_to_next_candidate(tmp_path: Path):
    configured_root = tmp_path / "configured-hf-home"
    configured_root.mkdir()
    (configured_root / "token").write_bytes(b"\xff\xfe")
    cache_root = tmp_path / ".cache" / "huggingface"
    cache_root.mkdir(parents=True)
    (cache_root / "token").write_text("cache-token", encoding="utf-8")

    provider = HuggingFaceTokenProvider(
        {"HF_HOME": str(configured_root)},
        home_dir=tmp_path,
    )

    assert provider.get_hf_token() == "cache-token"


def test_invalid_utf8_token_file_returns_none_when_no_candidate_is_valid(
    tmp_path: Path,
):
    cache_root = tmp_path / ".cache" / "huggingface"
    cache_root.mkdir(parents=True)
    (cache_root / "token").write_bytes(b"\x80")

    provider = HuggingFaceTokenProvider({}, home_dir=tmp_path)

    assert provider.get_hf_token() is None


def test_unexpandable_hf_home_is_a_configuration_error(tmp_path: Path):
    provider = HuggingFaceTokenProvider(
        {"HF_HOME": "~__dgx_vllm_launcher_missing_user__"},
        home_dir=tmp_path,
    )

    with pytest.raises(ConfigurationError, match="HF_HOME"):
        provider.get_hf_token()


def test_empty_hf_home_falls_through_to_default_token_file(tmp_path: Path):
    cache_root = tmp_path / ".cache" / "huggingface"
    cache_root.mkdir(parents=True)
    (cache_root / "token").write_text("cache-token", encoding="utf-8")

    provider = HuggingFaceTokenProvider(
        {"HF_HOME": "   "},
        home_dir=tmp_path,
    )

    assert provider.get_hf_token() == "cache-token"


def test_token_provider_returns_none_when_unconfigured(tmp_path: Path):
    assert HuggingFaceTokenProvider({}, home_dir=tmp_path).get_hf_token() is None
