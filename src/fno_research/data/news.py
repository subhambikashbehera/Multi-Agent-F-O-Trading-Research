"""Market news from public RSS feeds (no API key needed)."""

from __future__ import annotations

import logging
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

from fno_research.data.web import WebClient
from fno_research.models import NewsItem

log = logging.getLogger(__name__)

DEFAULT_FEEDS = {
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol Markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "LiveMint Markets": "https://www.livemint.com/rss/markets",
}
# Google News search, restricted to the last day, per underlying.
GOOGLE_NEWS_QUERIES = {
    "NIFTY": "Nifty OR Sensex OR \"stock market\" India",
    "BANKNIFTY": "\"Bank Nifty\" OR \"banking stocks\" OR RBI",
    "FINNIFTY": "\"Fin Nifty\" OR NBFC OR \"financial stocks\" India",
    "MIDCPNIFTY": "\"midcap stocks\" OR \"Nifty Midcap\"",
}


def google_news_url(underlying: str) -> str:
    query = GOOGLE_NEWS_QUERIES.get(underlying.upper(), underlying) + " when:1d"
    return ("https://news.google.com/rss/search?"
            + urllib.parse.urlencode({"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}))


def parse_rss(xml_text: str, source: str) -> list[NewsItem]:
    root = ET.fromstring(xml_text)
    items = []
    for node in root.iter("item"):
        published = node.findtext("pubDate")
        try:
            when = parsedate_to_datetime(published) if published else None
        except (TypeError, ValueError):
            when = None
        # Google News puts the publisher in <source>; fall back to the feed name.
        publisher = (node.findtext("source") or "").strip()
        items.append(
            NewsItem(
                title=(node.findtext("title") or "").strip(),
                summary=(node.findtext("description") or "").strip()[:500],
                source=publisher or source,
                published=when,
                url=(node.findtext("link") or "").strip(),
            )
        )
    return items


class RSSNewsProvider:
    def __init__(self, feeds: dict[str, str] | None = None, google_news: bool = True,
                 web: WebClient | None = None):
        self.feeds = feeds if feeds is not None else DEFAULT_FEEDS
        self.google_news = google_news
        self.web = web or WebClient(min_interval=0.3)

    def headlines(self, underlying: str, limit: int = 25) -> list[NewsItem]:
        feeds = dict(self.feeds)
        if self.google_news:
            feeds["Google News"] = google_news_url(underlying)
        items: list[NewsItem] = []
        for source, url in feeds.items():
            try:
                items.extend(parse_rss(self.web.get_text(url), source))
            except Exception as exc:  # one dead feed should not sink the run
                log.info("Feed %s failed: %s", source, exc)
        seen, unique = set(), []
        for item in items:
            key = item.title.lower()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(item)
        unique.sort(key=lambda i: i.published.timestamp() if i.published else 0, reverse=True)
        return unique[:limit]
