from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from fno_research.config import SwingSettings
from fno_research.data.nse import parse_lot_sizes
from fno_research.paper import PaperBook
from fno_research.swing.book import APPROVED, BLOCKED, PENDING, SwingBook
from fno_research.swing.data import SampleSwingData, sample_universe
from fno_research.swing.models import LONG, SHORT, Stock
from fno_research.swing.scanner import (
    SwingScanner,
    build_plan,
    build_short,
    check_plan,
    company_query,
    score_stock,
)
from fno_research.swing.universe import parse_constituents


def trend_frame(drift, n=300, base=1000.0, vol=0.01, volume=2e6, seed=1):
    rng = np.random.default_rng(seed)
    close = base * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    idx = pd.bdate_range(end="2026-09-22", periods=n)
    opens = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, close) * 1.005,
                         "low": np.minimum(opens, close) * 0.995, "close": close,
                         "volume": volume}, index=idx)


BENCH = trend_frame(0.0003, seed=2, base=25_000, vol=0.006)
STOCK = Stock(symbol="ABC", name="Abc Industries Ltd.", industry="Capital Goods")


# -- parsers ------------------------------------------------------------------------------

def test_parse_constituents():
    text = ("﻿Company Name,Industry,Symbol,Series,ISIN Code\n"
            "Tata Motors Ltd.,Automobile and Auto Components,TATAMOTORS,EQ,INE155A01022\n"
            "Some Bond,Financial Services,XYZ,N1,INE000\n")
    stocks = parse_constituents(text)
    assert [s.symbol for s in stocks] == ["TATAMOTORS"]
    assert stocks[0].industry.startswith("Automobile")


def test_parse_lot_sizes():
    text = ("UNDERLYING,SYMBOL,SEP-26,OCT-26,NOV-26\n"
            "Derivatives on Individual Securities,,,,\n"
            "TATA MOTORS LIMITED,TATAMOTORS,  800,800,800\n"
            "NIFTY 50,NIFTY,65,65,65\n")
    assert parse_lot_sizes(text) == {"TATAMOTORS": 800, "NIFTY": 65}


def test_company_query():
    assert company_query("Tata Motors Ltd.") == '"Tata Motors"'
    assert company_query("Infosys Limited") == '"Infosys"'
    assert company_query("") == ""


# -- scoring and plans ------------------------------------------------------------------

def test_uptrend_qualifies_long():
    s = score_stock(STOCK, trend_frame(0.003), BENCH, SwingSettings())
    assert s.side == LONG and s.qualified and s.above_200 and s.rs_60 > 0


def test_downtrend_needs_fno_to_short():
    cfg = SwingSettings()
    frame = trend_frame(-0.003)
    no_fno = score_stock(STOCK, frame, BENCH, cfg, fno=False)
    assert not no_fno.qualified and "not in F&O" in no_fno.reason
    fno = score_stock(STOCK, frame, BENCH, cfg, fno=True)
    assert fno.side == SHORT and fno.qualified
    cfg.allow_short = False
    assert not score_stock(STOCK, frame, BENCH, cfg, fno=True).qualified


def test_illiquid_stock_is_filtered():
    s = score_stock(STOCK, trend_frame(0.003, volume=1_000), BENCH, SwingSettings())
    assert not s.qualified and "traded value" in s.reason


def test_build_plan_risk_and_size():
    cfg = SwingSettings(capital=500_000, risk_per_trade_pct=1.0, max_position_pct=20.0)
    frame = trend_frame(0.003)
    plan = build_plan("ABC", frame, cfg)
    assert plan.stop < plan.entry < plan.target
    assert plan.reward_risk == pytest.approx(2.0, abs=0.02)
    assert plan.risk_amount <= 5_000 + plan.risk_per_share
    assert plan.position_value <= 100_000 + plan.entry
    assert plan.entry - plan.stop >= plan.atr * 0.99  # never tighter than 1 ATR


def test_check_plan_book_rules():
    cfg = SwingSettings()
    frame = trend_frame(0.003)
    s = score_stock(STOCK, frame, BENCH, cfg)
    plan = build_plan("ABC", frame, cfg)
    assert check_plan(plan, s, cfg, [], []).approved
    names = lambda d: {c.name for c in d.failures}  # noqa: E731
    assert "Not already held" in names(check_plan(plan, s, cfg, ["ABC"], ["x"]))
    assert "Industry concentration" in names(
        check_plan(plan, s, cfg, ["P", "Q"], ["Capital Goods", "Capital Goods"]))
    assert "Open positions" in names(check_plan(plan, s, cfg, list("ABCDE"), [""] * 5))
    assert "Kill switch" in names(check_plan(plan, s, cfg, [], [], {"active": True}))


def test_build_short_fits_budget_when_capital_allows():
    data = SampleSwingData()
    lots = data.lot_sizes()
    sym = "DEMO09"
    chain = data.option_chain(sym, lots[sym])
    s = score_stock(Stock(symbol=sym), data.daily(sym), data.benchmark(), SwingSettings(),
                    fno=True)
    small = build_short(s, chain, SwingSettings(capital=500_000))
    big = build_short(s, chain, SwingSettings(capital=2_000_000))
    assert small.strategy == "Bear put spread" and big.strategy == "Bear put spread"
    assert all(leg.option_type == "PE" for leg in big.legs)
    assert big.max_loss <= 20_000
    # With a small budget it falls back to the narrowest spread.
    width = lambda i: i.legs[0].strike - i.legs[1].strike  # noqa: E731
    assert width(small) <= width(big)


# -- book ---------------------------------------------------------------------------------

def make_idea(tmp_path, drift=0.003):
    cfg = SwingSettings()
    frame = trend_frame(drift)
    s = score_stock(STOCK, frame, BENCH, cfg)
    plan = build_plan("ABC", frame, cfg)
    from fno_research.swing.models import SwingIdea

    idea = SwingIdea(id="i1", created_at=datetime(2026, 9, 22), as_of=frame.index[-1].date(),
                     side=LONG, stock=s, plan=plan, risk=check_plan(plan, s, cfg, [], []))
    book = SwingBook(tmp_path / "s.db")
    book.add_ideas([idea])
    return book, idea, frame


def future_bars(frame, rows):
    idx = pd.bdate_range(start=frame.index[-1] + timedelta(days=1), periods=len(rows))
    new = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    new["volume"] = 1e6
    return pd.concat([frame, new])


def test_approve_opens_long_and_stop_exits(tmp_path):
    book, idea, frame = make_idea(tmp_path)
    assert book.ideas(PENDING)[0]["id"] == "i1"
    pid = book.decide("i1", True, "ok")
    assert book.ideas(APPROVED) and book.positions("open")[0]["id"] == pid
    p = idea.plan
    bars = future_bars(frame, [(p.entry, p.entry + 1, p.entry - 1, p.entry + 0.5),
                               (p.entry, p.entry + 1, p.stop - 1, p.stop + 1)])
    assert book.update({"ABC": bars}) == 1
    closed = book.positions("closed")[0]
    assert closed["exit_reason"] == "stop" and closed["exit_price"] == pytest.approx(p.stop)
    assert closed["pnl"] == pytest.approx((p.stop - p.entry) * p.qty, abs=0.01)
    # Daily P&L across both days adds up to the realized loss.
    total = sum(book.daily_pnl(d.date()) for d in bars.index[-2:])
    assert total == pytest.approx(closed["pnl"], abs=0.01)


@pytest.mark.parametrize("bar, reason", [
    ("gap_down", "gap below stop"), ("target", "target"), ("both", "stop"),
    ("gap_up", "gap above target"),
])
def test_exit_rules(tmp_path, bar, reason):
    book, idea, frame = make_idea(tmp_path)
    book.decide("i1", True)
    p = idea.plan
    rows = {
        "gap_down": (p.stop - 5, p.stop - 2, p.stop - 10, p.stop - 4),
        "target": (p.entry, p.target + 1, p.entry - 1, p.target),
        "both": (p.entry, p.target + 1, p.stop - 1, p.entry),
        "gap_up": (p.target + 5, p.target + 8, p.target + 2, p.target + 6),
    }[bar]
    book.update({"ABC": future_bars(frame, [rows])})
    assert book.positions("closed")[0]["exit_reason"] == reason


def test_time_stop(tmp_path):
    book, idea, frame = make_idea(tmp_path)
    book.decide("i1", True)
    p = idea.plan
    flat = [(p.entry, p.entry + 1, p.entry - 1, p.entry)] * p.time_stop_sessions
    book.update({"ABC": future_bars(frame, flat)})
    closed = book.positions("closed")[0]
    assert closed["exit_reason"] == "time stop"
    assert closed["sessions_held"] == p.time_stop_sessions


def test_update_is_idempotent(tmp_path):
    book, idea, frame = make_idea(tmp_path)
    book.decide("i1", True)
    p = idea.plan
    bars = future_bars(frame, [(p.entry, p.entry + 1, p.entry - 1, p.entry + 2)])
    book.update({"ABC": bars})
    book.update({"ABC": bars})
    pos = book.positions("open")[0]
    assert pos["sessions_held"] == 1 and pos["last_close"] == pytest.approx(p.entry + 2)


def test_decide_twice_fails(tmp_path):
    book, _, _ = make_idea(tmp_path)
    book.decide("i1", False, "no")
    with pytest.raises(ValueError):
        book.decide("i1", True)


# -- scanner ------------------------------------------------------------------------------

def scanner(tmp_path, cfg=None, **kw):
    cfg = cfg or SwingSettings()
    book, paper = SwingBook(tmp_path / "s.db"), PaperBook(tmp_path / "s.db")
    return SwingScanner(cfg, kw.pop("data", SampleSwingData()), sample_universe(), book=book,
                        paper=paper, ban_list=set, **kw), book, paper


def test_scan_produces_ranked_two_sided_ideas(tmp_path):
    sc, book, _ = scanner(tmp_path)
    report = sc.run()
    assert len(report.scored) == 40 and not report.failed
    sides = {i.side for i in report.ideas}
    assert sides == {LONG, SHORT}
    approved = [i for i in report.ideas if i.risk.approved]
    assert 0 < len(approved) <= SwingSettings().max_open
    strengths = [abs(i.stock.score) for i in report.ideas]
    assert strengths[: len(approved)] == sorted(strengths, reverse=True)[: len(approved)]
    book.add_ideas(report.ideas)
    assert len(book.ideas(PENDING)) == len(approved)
    assert len(book.ideas(BLOCKED)) == len(report.ideas) - len(approved)


def test_short_approval_opens_paper_spread(tmp_path):
    cfg = SwingSettings(capital=2_000_000)  # big enough for one stock-option spread
    sc, book, paper = scanner(tmp_path, cfg)
    report = sc.run()
    book.add_ideas(report.ideas)
    short = next(i for i in report.ideas if i.side == SHORT and i.risk.approved)
    pid = book.decide(short.id, True, paper=paper)
    pos = paper.positions("open")[0]
    assert pos["id"] == pid and pos["underlying"] == short.symbol
    # The next scan re-prices it from the stock's option chain (sample chain time).
    sc.run()
    assert paper.positions("open")[0]["last_marked_at"] == "2026-09-22T15:30:00"
    marks = paper.conn.execute("SELECT COUNT(*) FROM marks WHERE position_id = ?",
                               (pid,)).fetchone()[0]
    assert marks == 1


class FakeNews:
    def __init__(self, score):
        self.score = score
        self.calls = []

    def analyse(self, symbol, query=None):
        from fno_research.agents.base import make_signal

        self.calls.append((symbol, query))
        return make_signal("news", self.score, 0.9, "Auditor resigned; results delayed.", {},
                           "context")


def test_bad_news_turns_held_long_into_exit_alert(tmp_path):
    sc, book, _ = scanner(tmp_path)
    report = sc.run()
    book.add_ideas(report.ideas)
    top = next(i for i in report.ideas if i.side == LONG and i.risk.approved)
    pid = book.decide(top.id, True)

    # Same prices, but now every shortlisted stock's news is terrible.
    sc2, _, _ = scanner(tmp_path, news_agent=FakeNews(-1.0))
    sc2.book, sc2.paper = book, sc.paper
    report2 = sc2.run()
    assert top.symbol in {sym for sym, _ in sc2.news_agent.calls}
    held = next(s for s in report2.scored if s.symbol == top.symbol)
    assert held.agents["news"]["score"] == -1.0 and held.score < top.stock.score
    alerts = [a for a in report2.exit_alerts if a.position_id == pid]
    # The chart is still bullish, but news this bad raises an exit alert by itself.
    assert held.score > 0
    assert len(alerts) == 1 and alerts[0].reason == "Strongly negative news"


def test_mild_news_does_not_alert(tmp_path):
    sc, book, _ = scanner(tmp_path)
    report = sc.run()
    book.add_ideas(report.ideas)
    top = next(i for i in report.ideas if i.side == LONG and i.risk.approved)
    book.decide(top.id, True)
    sc2, _, _ = scanner(tmp_path, news_agent=FakeNews(-0.3))
    sc2.book, sc2.paper = book, sc.paper
    assert not sc2.run().exit_alerts


def test_trend_reversal_raises_exit_alert(tmp_path):
    data = SampleSwingData()
    sc, book, _ = scanner(tmp_path, data=data)
    report = sc.run()
    book.add_ideas(report.ideas)
    top = next(i for i in report.ideas if i.side == LONG and i.risk.approved)
    pid = book.decide(top.id, True)
    # The held stock rolls over hard afterwards.
    frame = data.daily(top.symbol)
    crash = trend_frame(-0.02, n=40, base=float(frame.close.iloc[-1]), seed=9)
    crash.index = pd.bdate_range(start=frame.index[-1] + timedelta(days=1), periods=40)
    data.overrides[top.symbol] = pd.concat([frame, crash])
    report2 = sc.run()
    closed = [p for p in book.positions("closed") if p["id"] == pid]
    alerts = [a for a in report2.exit_alerts if a.position_id == pid]
    # Either the stop took it out or, if still open, the scan flags it for exit.
    assert closed or alerts
    if closed:
        assert closed[0]["exit_reason"] in ("stop", "gap below stop")
