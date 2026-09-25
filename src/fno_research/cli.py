"""Command-line helpers: daily Kite login and a headless research run."""

from __future__ import annotations

import argparse

from fno_research.config import Settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fno-research")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("kite-login", help="Get today's Kite access token")

    run = sub.add_parser("run", help="Run the pipeline once and queue the report")
    run.add_argument("underlying", nargs="?", default="NIFTY")
    run.add_argument("--sample", action="store_true", help="Use synthetic data (no Kite)")

    args = parser.parse_args(argv)
    settings = Settings()

    if args.command == "kite-login":
        from fno_research.data.kite import exchange_request_token, login_url

        print("1. Open this URL and log in:\n  ", login_url(settings))
        print("2. Copy the request_token parameter from the redirect URL.")
        token = exchange_request_token(settings, input("request_token: ").strip())
        print(f"\nAdd this to .env (valid until tomorrow morning):\nKITE_ACCESS_TOKEN={token}")
        return

    from fno_research.factory import build_pipeline
    from fno_research.review import ReviewQueue

    pipeline = build_pipeline(settings, sample=args.sample)
    report = pipeline.run(args.underlying)
    status = ReviewQueue(settings.db_path).add(report)
    print(report.summary)
    for s in report.signals:
        print(f"  {s.agent:<20} {s.score:+.2f}  conf {s.confidence:.2f}  {s.rationale}")
    print(f"Report {report.id} saved with status {status}.")


if __name__ == "__main__":
    main()
