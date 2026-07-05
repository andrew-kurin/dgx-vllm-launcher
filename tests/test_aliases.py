from __future__ import annotations

import sys

from dgx_vllm_launcher import variant_aliases


def test_qwen_fp8_alias_prepends_variant(monkeypatch):
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 42

    monkeypatch.setattr(variant_aliases, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["qwen-fp8", "--no-warmup", "--reasoning"])

    code = variant_aliases.qwen_fp8()

    assert code == 42
    assert captured["argv"] == ["fp8", "--no-warmup", "--reasoning"]


def test_qwen_nvfp4_alias_prepends_variant(monkeypatch):
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 7

    monkeypatch.setattr(variant_aliases, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["qwen-nvfp4", "--detach"])

    code = variant_aliases.qwen_nvfp4()

    assert code == 7
    assert captured["argv"] == ["nvfp4", "--detach"]


def test_gemma4_alias_prepends_variant(monkeypatch):
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 9

    monkeypatch.setattr(variant_aliases, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["gemma4-nvfp4", "--restart-policy", "unless-stopped"])

    code = variant_aliases.gemma4_nvfp4()

    assert code == 9
    assert captured["argv"] == ["gemma4-nvfp4", "--restart-policy", "unless-stopped"]
