"""Market news from public RSS feeds (no API key needed)."""

from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

from fno_research.models import NewsItem

DEFAULT_FEEDS = {
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol Markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "LiveMint Markets": "https://www.livemint.com/rss/markets",
}


class RSSNewsProvider:
    def __init__(self, feeds: dict[str, str] | None = None, timeout: float = 10.0):
        self.feeds = feeds or DEFAULT_FEEDS
        self.timeout = timeout

    def _fetch(self, source: str, url: str) -> list[NewsItem]:
        request = urllib.request.Request(url, headers={"User-Agent": "fno-research/0.1"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            root = ET.fromstring(response.read())
        items = []
        for node in root.iter("item"):
            published = node.findtext("pubDate")
            try:
                when = parsedate_to_datetime(published) if published else None
            except (TypeError, ValueError):
                when = None
            items.append(
                NewsItem(
                    title=(node.findtext("title") or "").strip(),
                    summary=(node.findtext("description") or "").strip()[:500],
                    source=source,
                    published=when,
                    url=(node.findtext("link") or "").strip(),
                )
            )
        return items

    def headlines(self, underlying: str, limit: int = 25) -> list[NewsItem]:
        items: list[NewsItem] = []
        for source, url in self.feeds.items():
            try:
                items.extend(self._fetch(source, url))
            except Exception:  # one dead feed should not sink the run
                continue
        items.sort(key=lambda i: i.published.timestamp() if i.published else 0, reverse=True)
        return items[:limit]
