"""Scripted multi-day session on synthetic data, with a scripted reviewer.

Plays out: a bullish confluence idea that gets approved and paper-traded, a
technical-vs-context conflict that must be vetoed, a mark-to-market gain, a crash that
trips the daily-loss kill switch, and settlement at expiry. Prints what the system did at
each step and exits non-zero if any expectation fails.

    python scripts/scenario_session.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

from fno_research.config import Settings
from fno_research.data.sample import SampleDataProvider, StaticNewsProvider
from fno_research.models import FlowSnapshot
from fno_research.paper import PaperBook
from fno_research.pipeline import ResearchPipeline
from fno_research.review import BLOCKED, PENDING, ReviewQueue, decision_log
from fno_research.store import FeatureStore

failures: list[str] = []


def expect(ok: bool, what: str) -> None:
    print(f"    {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)


class Flows:
    def __init__(self, day: date, fii: float, dii: float):
        self.snap = FlowSnapshot(date=day, fii_net=fii, dii_net=dii)

    def fii_dii(self):
        return self.snap


def run(settings, store, book, queue, underlying, when, spot, drift, fii, dii=0.0):
    market = SampleDataProvider(drift=drift, as_of=when, spot=spot)
    pipeline = ResearchPipeline(settings, market, StaticNewsProvider(),
                                flows=Flows(when.date(), fii, dii), store=store, paper=book)
    report = pipeline.run(underlying)
    status = queue.add(report)
    agg = report.aggregate
    groups = ", ".join(f"{g} {gs.score:+.2f}/{gs.confidence:.2f}" for g, gs in agg.groups.items())
    print(f"  {when:%a %d %b %H:%M} {underlying} spot {spot:,.0f} -> {status}")
    print(f"    groups: {groups}; multiplier {agg.allocation_multiplier:.2f}")
    print(f"    {report.summary}")
    return report, status


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    settings = Settings()
    settings.db_path = tmp / "session.db"
    settings.risk.capital = 1_000_000
    settings.risk.daily_loss_pct = 0.25  # ₹2,500: tight, so one lot can trip it
    store, book, queue = (FeatureStore(settings.db_path), PaperBook(settings.db_path),
                          ReviewQueue(settings.db_path))
    print(f"Session DB: {settings.db_path}\n")

    print("Day 1 — NIFTY uptrend with FII buying")
    d1 = datetime(2026, 9, 22, 11, 0)
    for day in range(15, 19):  # four earlier sessions of flows already in the store
        store.save_flows(FlowSnapshot(date=date(2026, 9, day), fii_net=2_500, dii_net=300))
    report, status = run(settings, store, book, queue, "NIFTY", d1, 25_000, 0.004, 2_500)
    expect(status == PENDING, "bullish confluence idea passes risk")
    expect(report.aggregate.allocation_multiplier > 0.5, "confluence gets a large multiplier")
    if status == PENDING:
        approved = queue.decide(report.id, True, "Operator: trend + flows aligned, 1 lot.")
        pid = book.open_from_report(approved, when=d1)
        print(f"    Reviewer approved -> paper position #{pid}")

    print("\nDay 1 — BANKNIFTY downtrend but heavy FII buying (conflict)")
    for day in range(15, 19):
        store.save_flows(FlowSnapshot(date=date(2026, 9, day), fii_net=9_000, dii_net=300))
    report, status = run(settings, store, book, queue, "BANKNIFTY", d1, 55_000, -0.004, 9_000)
    expect(report.aggregate.veto, "conflicting groups are vetoed")
    expect(report.idea is None, "vetoed view produces no trade")

    print("\nDay 2 — NIFTY rallies to 25,150")
    d2 = datetime(2026, 9, 23, 14, 0)
    report, status = run(settings, store, book, queue, "NIFTY", d2, 25_150, 0.004, 2_000)
    pos = book.positions("open")[0]
    print(f"    Paper #{pos['id']} unrealized ₹{pos['unrealized_pnl']:+,.0f}")
    expect(pos["unrealized_pnl"] > 0, "open bull spread gains on a rally")
    failed = {c.name for c in report.risk.failures} if report.risk else set()
    expect(status != PENDING and "Existing exposure" in failed,
           "a second NIFTY idea is blocked while one is open")
    if status == PENDING:
        queue.decide(report.id, False, "Operator: already long NIFTY.")

    print("\nDay 3 — NIFTY gaps down to 24,600")
    d3 = datetime(2026, 9, 24, 11, 0)
    report, status = run(settings, store, book, queue, "NIFTY", d3, 24_600, -0.004, -3_000)
    pnl = book.daily_pnl(d3.date())
    state = book.kill_switch()
    print(f"    Day P&L ₹{pnl:+,.0f}; kill switch {'TRIPPED' if state['active'] else 'off'}")
    expect(state["active"], "kill switch trips on breaching the daily loss limit")

    print("\nDay 3 — BANKNIFTY sell-off with FII selling, while the kill switch is active")
    for day in range(18, 24):  # a week of heavy FII selling: technical and context agree
        store.save_flows(FlowSnapshot(date=date(2026, 9, day), fii_net=-9_000, dii_net=500))
    report, status = run(settings, store, book, queue, "BANKNIFTY", d3.replace(hour=12),
                         55_000, -0.004, -9_000, dii=500)
    failed = {c.name for c in report.risk.failures} if report.risk else set()
    print(f"    Risk failures: {sorted(failed)}")
    expect(report.idea is not None, "aligned bearish view produces an idea")
    expect(status == BLOCKED and "Kill switch" in failed,
           "the idea is blocked by the tripped kill switch")
    book.reset_kill_switch()
    print("    Operator reviewed the loss and reset the kill switch.")

    print("\nExpiry — Tue 29 Sep 15:35, NIFTY settles at 25,300")
    expiry_close = datetime(2026, 9, 29, 15, 35)
    book.mark(SampleDataProvider(as_of=expiry_close, spot=25_300).option_chain("NIFTY"))
    closed = book.positions("closed")
    expect(len(closed) == 1 and not book.positions("open"), "position settles at expiry")
    if closed:
        p = closed[0]
        print(f"    #{p['id']} settled: entry ₹{p['entry_cost']:,.0f}, exit "
              f"₹{p['exit_value']:,.0f}, realized ₹{p['realized_pnl']:+,.0f}")

    approved = [r for r in decision_log(queue, book) if r["paper"]]
    print(f"    Decision log outcomes: {[(r['status'], r['paper'], r['pnl']) for r in approved]}")
    expect(len(approved) == 1 and approved[0]["paper"] == "closed"
           and abs(approved[0]["pnl"] - closed[0]["realized_pnl"]) < 0.01,
           "decision log shows the approved idea's realized outcome")
    runs = len(queue.list(limit=1000))
    print(f"\nLogged {runs} runs; feature store has {len(store.chains('NIFTY'))} NIFTY chains.")
    print(f"\n{'ALL EXPECTATIONS MET' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
