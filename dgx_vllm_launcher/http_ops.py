from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class HttpResult:
    status: int | None
    body: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300

    def failure_detail(self) -> str:
        details = []
        if self.status is not None:
            details.append(f"HTTP {self.status}")
        if self.error:
            details.append(self.error)
        if self.body:
            details.append(self.body.strip())
        return ": ".join(detail for detail in details if detail) or "request failed"


class VllmClient:
    def __init__(
        self,
        base_url: str,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self._health_url = f"{base_url.rstrip('/')}/health"
        self._completion_url = f"{base_url.rstrip('/')}/v1/completions"
        self._opener = opener

    def health(self, *, timeout: float) -> bool:
        try:
            with self._opener(self._health_url, timeout=timeout) as response:
                status = cast(int, response.status)
                return 200 <= status < 300
        except Exception:
            return False

    def completion(
        self,
        payload: Mapping[str, object],
        *,
        timeout: float,
    ) -> HttpResult:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._completion_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with self._opener(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                return HttpResult(status=cast(int, response.status), body=body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return HttpResult(status=exc.code, body=body, error=str(exc))
        except Exception as exc:
            return HttpResult(status=None, body=None, error=str(exc))
