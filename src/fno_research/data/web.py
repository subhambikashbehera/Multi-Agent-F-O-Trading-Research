"""HTTP session for scraping public sites: browser-like headers, cookies, retries, caching."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


class WebClient:
    """Small wrapper over requests.Session.

    - `warmup_url` is fetched first (and again after a 401/403) to collect cookies; NSE's
      API refuses requests that lack the cookies its home page sets.
    - Responses are cached for `cache_ttl` seconds so a dashboard refresh does not hammer
      the site, and requests are spaced at least `min_interval` seconds apart.
    """

    def __init__(self, warmup_url: str | None = None, referer: str | None = None,
                 timeout: float = 15.0, cache_ttl: float = 60.0, min_interval: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update(BROWSER_HEADERS)
        if referer:
            self.session.headers["Referer"] = referer
        self.warmup_url = warmup_url
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.min_interval = min_interval
        self._warmed = False
        self._last_request = 0.0
        self._cache: dict[str, tuple[float, Any]] = {}

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _warmup(self) -> None:
        if self.warmup_url:
            self._throttle()
            self.session.get(self.warmup_url, timeout=self.timeout,
                             headers={"Accept": "text/html,*/*"})
        self._warmed = True

    def get_json(self, url: str, params: dict | None = None) -> Any:
        key = url + repr(sorted((params or {}).items()))
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < self.cache_ttl:
            return cached[1]
        if not self._warmed:
            self._warmup()
        for attempt in range(2):
            self._throttle()
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code in (401, 403) and attempt == 0:
                log.info("Got %s from %s; refreshing cookies", response.status_code, url)
                self._warmup()
                continue
            response.raise_for_status()
            data = response.json()
            self._cache[key] = (time.monotonic(), data)
            return data
        raise RuntimeError(f"Could not fetch {url}")

    def get_text(self, url: str, params: dict | None = None) -> str:
        self._throttle()
        response = self.session.get(url, params=params, timeout=self.timeout,
                                    headers={"Accept": "application/rss+xml,text/xml,*/*"})
        response.raise_for_status()
        return response.text
