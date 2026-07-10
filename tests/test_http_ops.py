from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

from dgx_vllm_launcher.http_ops import VllmClient


class FakeResponse:
    def __init__(self, status: int, body: str = "") -> None:
        self.status = status
        self._body = body.encode()

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_health_uses_configured_base_url_and_timeout():
    captured: dict[str, Any] = {}

    def opener(url, *, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse(204)

    client = VllmClient("http://127.0.0.1:9000/", opener=opener)

    assert client.health(timeout=1.25) is True
    assert captured == {
        "url": "http://127.0.0.1:9000/health",
        "timeout": 1.25,
    }


def test_completion_returns_typed_success_result():
    captured: dict[str, Any] = {}

    def opener(request, *, timeout):
        assert isinstance(request, urllib.request.Request)
        captured["url"] = request.full_url
        assert isinstance(request.data, bytes)
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(200, '{"id":"ok"}')

    result = VllmClient("http://localhost:8000", opener=opener).completion(
        {"model": "model", "prompt": "hello"},
        timeout=12,
    )

    assert result.ok is True
    assert result.body == '{"id":"ok"}'
    assert captured["url"] == "http://localhost:8000/v1/completions"
    assert captured["payload"]["prompt"] == "hello"
    assert captured["timeout"] == 12


def test_completion_preserves_http_error_body():
    def opener(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "http://localhost/v1/completions",
            503,
            "unavailable",
            Message(),
            io.BytesIO(b'{"error":"loading"}'),
        )

    result = VllmClient("http://localhost", opener=opener).completion(
        {"model": "model"},
        timeout=1,
    )

    assert result.ok is False
    assert result.status == 503
    assert result.body == '{"error":"loading"}'
    assert "loading" in result.failure_detail()


def test_completion_reports_transport_error():
    def opener(*_args, **_kwargs):
        raise TimeoutError("too slow")

    result = VllmClient("http://localhost", opener=opener).completion(
        {"model": "model"},
        timeout=1,
    )

    assert result.ok is False
    assert result.status is None
    assert "too slow" in result.failure_detail()
