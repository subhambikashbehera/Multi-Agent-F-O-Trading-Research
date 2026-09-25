"""Index constituents from niftyindices.com / NSE archives, cached locally.

If both downloads fail, put the CSV (as downloaded from niftyindices.com) at
data/cache/universe_<name>.csv and it will be used.
"""

from __future__ import annotations

import csv
import io
import logging
import time
from pathlib import Path

from fno_research.data.web import WebClient
from fno_research.swing.models import Stock

log = logging.getLogger(__name__)
SOURCES = [
    "https://nsearchives.nseindia.com/content/indices/ind_{name}list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_{name}list.csv",
]
MAX_AGE_DAYS = 7  # constituents change at semi-annual rebalances


def parse_constituents(text: str) -> list[Stock]:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    stocks = []
    for row in reader:
        row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
        symbol = row.get("symbol")
        if symbol and row.get("series", "EQ") in ("EQ", "BE", ""):
            stocks.append(Stock(symbol=symbol.upper(), name=row.get("company name", ""),
                                industry=row.get("industry", "")))
    return stocks


def load_universe(name: str, cache_dir: Path = Path("data/cache"),
                  web: WebClient | None = None) -> list[Stock]:
    path = cache_dir / f"universe_{name}.csv"
    fresh = path.exists() and time.time() - path.stat().st_mtime < MAX_AGE_DAYS * 86400
    if not fresh:
        web = web or WebClient(warmup_url=None)
        for url in SOURCES:
            try:
                text = web.get_text(url.format(name=name))
                if parse_constituents(text):
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    path.write_text(text)
                    break
            except Exception as exc:
                log.info("Universe download from %s failed: %s", url, exc)
    if not path.exists():
        raise RuntimeError(
            f"Could not download the {name} constituent list. Save it from niftyindices.com "
            f"to {path} and run again."
        )
    return parse_constituents(path.read_text())
