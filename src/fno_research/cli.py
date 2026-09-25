"""Command-line entry points."""

from __future__ import annotations

import argparse

import pandas as pd

from fno_research.config import SUPPORTED_UNDERLYINGS, Settings
from fno_research.factory import SOURCES


def _kite_login(settings: Settings) -> None:
    from fno_research.data.kite import exchange_request_token, login_url

    print("1. Open this URL and log in:\n  ", login_url(settings))
    print("2. Copy the request_token parameter from the redirect URL.")
    token = exchange_request_token(settings, input("request_token: ").strip())
    print(f"\nAdd this to .env (valid until tomorrow morning):\nKITE_ACCESS_TOKEN={token}")


def _run(settings: Settings, underlying: str, source: str | None) -> None:
    from fno_research.factory import build_pipeline
    from fno_research.review import ReviewQueue

    report = build_pipeline(settings, source).run(underlying)
    status = ReviewQueue(settings.db_path).add(report)
    print(report.summary)
    for s in report.signals:
        print(f"  [{s.group:<9}] {s.agent:<20} {s.score:+.2f}  conf {s.confidence:.2f}  "
              f"{s.rationale}")
    if report.risk:
        for c in report.risk.failures:
            print(f"  ✗ {c.name}: {c.detail}")
    print(f"Report {report.id} saved with status {status}.")


def _backtest(settings: Settings, underlying: str, source: str | None, years: float) -> None:
    from fno_research.analytics.indicators import candles_to_frame
    from fno_research.backtest import backtest_technical
    from fno_research.factory import build_market

    market = build_market(settings, source)
    daily = candles_to_frame(market.candles(underlying, "day", int(365 * years) + 200))
    result = backtest_technical(daily, weights=settings.weights)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(f"{underlying}: {len(result.signals)} test days "
              f"({result.signals.index[0]:%d %b %Y} to {result.signals.index[-1]:%d %b %Y})")
        print(result.summary.to_string(index=False))


def _review(settings: Settings, action: str, report_id: str | None, note: str) -> None:
    from fno_research.paper import PaperBook
    from fno_research.review import PENDING, ReviewQueue

    queue = ReviewQueue(settings.db_path)
    if action == "list":
        for row in queue.list(PENDING):
            print(f"{row['id']}  {row['created_at'][:16]}  {row['report'].summary}")
        return
    report = queue.decide(report_id, approve=action == "approve", note=note)
    if action == "approve":
        pos = PaperBook(settings.db_path).open_from_report(report)
        print(f"Approved; paper position {pos} opened.")
    else:
        print("Rejected.")


def _paper(settings: Settings, action: str) -> None:
    from fno_research.paper import PaperBook

    book = PaperBook(settings.db_path)
    if action == "reset-kill-switch":
        book.reset_kill_switch()
        print("Kill switch reset.")
        return
    state = book.kill_switch()
    print("Kill switch:", "TRIPPED — " + state.get("reason", "") if state.get("active")
          else "off")
    for p in book.positions():
        pnl = p["realized_pnl"] if p["status"] == "closed" else p["unrealized_pnl"]
        print(f"#{p['id']} {p['status']:<6} {p['underlying']} {p['strategy']} exp "
              f"{p['expiry']:%d %b}  P&L ₹{pnl:+,.0f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fno-research")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("kite-login", help="Get today's Kite access token")

    run = sub.add_parser("run", help="Run the pipeline once and queue the report")
    run.add_argument("underlying", nargs="?", default="NIFTY", choices=SUPPORTED_UNDERLYINGS)
    run.add_argument("--source", choices=SOURCES, help="Market data (default FNO_DATA_SOURCE)")

    bt = sub.add_parser("backtest", help="Walk-forward test of the technical agents")
    bt.add_argument("underlying", nargs="?", default="NIFTY", choices=SUPPORTED_UNDERLYINGS)
    bt.add_argument("--source", choices=SOURCES)
    bt.add_argument("--years", type=float, default=3.0)

    rv = sub.add_parser("review", help="Work the review queue")
    rv.add_argument("action", choices=["list", "approve", "reject"])
    rv.add_argument("report_id", nargs="?")
    rv.add_argument("--note", default="")

    pp = sub.add_parser("paper", help="Paper positions and the kill switch")
    pp.add_argument("action", nargs="?", default="list", choices=["list", "reset-kill-switch"])

    args = parser.parse_args(argv)
    settings = Settings()
    if args.command == "kite-login":
        _kite_login(settings)
    elif args.command == "run":
        _run(settings, args.underlying, args.source)
    elif args.command == "backtest":
        _backtest(settings, args.underlying, args.source, args.years)
    elif args.command == "review":
        if args.action != "list" and not args.report_id:
            parser.error("report_id is required to approve or reject")
        _review(settings, args.action, args.report_id, args.note)
    elif args.command == "paper":
        _paper(settings, args.action)


if __name__ == "__main__":
    main()
