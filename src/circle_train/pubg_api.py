from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from circle_train.config import RAW_DIR


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self._min_interval = 60.0 / max(1, requests_per_minute)
        self._last_request_at = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        sleep_for = self._min_interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_request_at = time.monotonic()


class PubgApiClient:
    def __init__(
        self,
        api_key: str,
        shard: str = "steam",
        requests_per_minute: int = 10,
        cache_raw: bool | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("PUBG_API_KEY is required. Set it in .env or the environment.")

        self._api_key = api_key
        self._shard = shard
        self._base_url = f"https://api.pubg.com/shards/{shard}"
        self._session = requests.Session()
        self._limiter = RateLimiter(requests_per_minute)
        self._cache_raw = should_cache_raw() if cache_raw is None else cache_raw

    def get_samples(self, created_at_start: str | None = None) -> dict[str, Any]:
        params = {}
        if created_at_start:
            params["filter[createdAt-start]"] = created_at_start

        return self._get_json(f"{self._base_url}/samples", params=params, auth=True)

    def get_match(self, match_id: str, use_cache: bool = True) -> dict[str, Any]:
        cache_path = RAW_DIR / f"match_{match_id}.json"
        if self._cache_raw and use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        payload = self._get_json(f"{self._base_url}/matches/{match_id}", auth=False)
        if self._cache_raw:
            cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def get_telemetry(self, match_id: str, telemetry_url: str, use_cache: bool = True) -> list[dict[str, Any]]:
        cache_path = RAW_DIR / f"telemetry_{match_id}.json"
        if self._cache_raw and use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        self._limiter.wait()
        response = self._session.get(
            telemetry_url,
            headers={"Accept-Encoding": "gzip"},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if self._cache_raw:
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    def _get_json(self, url: str, params: dict[str, str] | None = None, auth: bool = True) -> dict[str, Any]:
        headers = {"Accept": "application/vnd.api+json"}
        if auth:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if auth:
            self._limiter.wait()
        response = self._session.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()


def should_cache_raw() -> bool:
    value = os.getenv("PUBG_CACHE_RAW", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}
