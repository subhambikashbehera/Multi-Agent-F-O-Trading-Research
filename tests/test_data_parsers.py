"""Parsers for the free sources, tested on responses shaped like the real ones."""

from datetime import date

import pytest

from fno_research.data.news import google_news_url, parse_rss
from fno_research.data.nse import (
    parse_all_indices,
    parse_fii_dii,
    parse_option_chain,
)
from fno_research.data.yahoo import parse_chart


def nse_leg(strike, opt, ltp, oi, iv=13.5):
    return {
        "strikePrice": strike, "expiryDate": "30-Sep-2026", "underlying": "NIFTY",
        "openInterest": oi, "changeinOpenInterest": oi // 10, "totalTradedVolume": 5000,
        "impliedVolatility": iv, "lastPrice": ltp, "bidprice": ltp - 0.5,
        "askPrice": ltp + 0.5, "underlyingValue": 25_012.4,
    }


def nse_chain_payload(expiries=("30-Sep-2026", "07-Oct-2026")):
    rows = []
    for exp in expiries:
        for strike in range(24_500, 25_550, 50):
            row = {"strikePrice": strike, "expiryDate": exp}
            ce = nse_leg(strike, "CE", max(25_012 - strike, 0) + 40, 1000 + strike % 700)
            pe = nse_leg(strike, "PE", max(strike - 25_012, 0) + 40, 1200 + strike % 500)
            ce["expiryDate"] = pe["expiryDate"] = exp
            row["CE"], row["PE"] = ce, pe
            rows.append(row)
    return {"records": {"expiryDates": list(expiries), "data": rows,
                        "timestamp": "25-Sep-2026 11:05:00", "underlyingValue": 25_012.4}}


def test_parse_nse_option_chain(monkeypatch):
    monkeypatch.setenv("FNO_LOT_NIFTY", "65")
    chain = parse_option_chain(nse_chain_payload(), "NIFTY", strikes_each_side=5)
    assert chain.expiry == date(2026, 9, 30)  # nearest expiry only
    assert chain.spot == pytest.approx(25_012.4)
    assert chain.lot_size == 65
    assert len(chain.strikes()) == 11 and chain.atm_strike() == 25_000
    q = chain.get(25_000, "CE")
    assert q.oi == (1000 + 25_000 % 700) * 65  # contracts -> units
    assert q.iv == pytest.approx(0.135)
    assert q.bid < q.last_price < q.ask
    assert chain.as_of.hour == 11


def test_parse_nse_chain_with_explicit_expiry():
    chain = parse_option_chain(nse_chain_payload(), "NIFTY", 5, expiry=date(2026, 10, 7))
    assert chain.expiry == date(2026, 10, 7)


def test_parse_nse_chain_rejects_empty():
    with pytest.raises(ValueError):
        parse_option_chain({"records": {"data": []}}, "NIFTY")


def test_parse_all_indices():
    payload = {"data": [{"index": "NIFTY 50", "last": 25012.4},
                        {"index": "INDIA VIX", "last": "13.25"},
                        {"index": "NIFTY BANK", "last": "55,120.10"}]}
    out = parse_all_indices(payload)
    assert out["NIFTY 50"] == 25012.4
    assert out["INDIA VIX"] == 13.25
    assert out["NIFTY BANK"] == 55120.10


def test_parse_fii_dii():
    payload = [
        {"category": "DII **", "date": "24-Sep-2026", "buyValue": "12000",
         "sellValue": "10500", "netValue": "1500.25"},
        {"category": "FII/FPI **", "date": "24-Sep-2026", "buyValue": "9000",
         "sellValue": "11000", "netValue": "-2000.5"},
    ]
    snap = parse_fii_dii(payload)
    assert snap.date == date(2026, 9, 24)
    assert snap.fii_net == -2000.5 and snap.dii_net == 1500.25
    assert parse_fii_dii([]) is None


def test_parse_yahoo_chart_skips_holes():
    payload = {"chart": {"result": [{
        "meta": {"symbol": "^NSEI"},
        "timestamp": [1790208000, 1790294400, 1790380800],
        "indicators": {"quote": [{
            "open": [25000, None, 25100], "high": [25100, None, 25200],
            "low": [24900, None, 25000], "close": [25050, None, 25150],
            "volume": [300000, None, 0],
        }]},
    }], "error": None}}
    candles = parse_chart(payload)
    assert [c.close for c in candles] == [25050, 25150]
    assert candles[0].volume == 300000


def test_parse_yahoo_error():
    with pytest.raises(ValueError):
        parse_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})


def test_parse_rss_google_news_source():
    xml = """<rss><channel>
      <item><title>Nifty ends higher as banks rally</title><link>https://x/1</link>
        <pubDate>Thu, 24 Sep 2026 10:00:00 GMT</pubDate><source url="https://et">ET</source>
      </item>
      <item><title>RBI holds rates</title><pubDate>not a date</pubDate></item>
    </channel></rss>"""
    items = parse_rss(xml, "Google News")
    assert items[0].source == "ET" and items[0].published.year == 2026
    assert items[1].source == "Google News" and items[1].published is None


def test_google_news_url_is_per_underlying():
    assert "Bank+Nifty" in google_news_url("BANKNIFTY")
    assert "when%3A1d" in google_news_url("NIFTY")
