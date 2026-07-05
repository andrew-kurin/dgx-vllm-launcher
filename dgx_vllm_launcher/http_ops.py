from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, cast


def health_ok(
    *,
    health_url: str = "http://127.0.0.1:8000/health",
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: int = 2,
) -> bool:
    try:
        with opener(health_url, timeout=timeout) as response:
            status = cast(int, response.status)
            return 200 <= status < 300
    except Exception:
        return False


def request_completion(
    payload: dict,
    *,
    endpoint: str = "http://127.0.0.1:8000/v1/completions",
    max_time: int = 120,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[int | None, str | None, Exception | None]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with opener(request, timeout=max_time) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, body, exc
    except Exception as exc:
        return None, None, exc
