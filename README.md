# Multi-Agent F&O Trading Research

A multi-agent research pipeline for Indian index futures and options. Technical and context
agents each form a view, an aggregator combines them (and vetoes when they conflict), a risk
engine sizes and checks the idea, and a person approves it before anything happens.

**This is a research tool. It never places real orders.** Approving an idea opens a *paper*
position.

## Where it stands against the plan

| # | Step | Status |
|---|------|--------|
| 1 | Data pipeline | **Built.** Free sources by default: NSE website (option chain, spot, India VIX, FII/DII) and Yahoo Finance (OHLCV history); Google News + ET/Moneycontrol/LiveMint RSS. Kite Connect provider ready for later. |
| 2 | Feature store | **Built.** SQLite: every run's agent features, VIX regime, expiry tag, FII/DII history, and a full option-chain snapshot. IV comes from NSE (or Black-Scholes when missing). |
| 3 | Technical agents | **Built + backtest harness.** Trend (MA cross, ADX, Supertrend), momentum (RSI, MACD), volatility (ATR, Bollinger), volume (OBV, VWAP). |
| 4 | Context agents | **Built.** Options positioning (PCR, OI walls, OI build-up, fresh OI, max pain, IV skew), FII/DII flows, news (Claude). Scored from the feature store as history accumulates. |
| 5 | Aggregator | **Built, weights unvalidated.** Technical/context groups, conflict veto, allocation multiplier. Weights and thresholds are placeholders until backtested on real data. |
| 6 | Risk engine | **Built.** Capital and margin checks, 1-lot cap while testing, F&O ban list, daily-loss kill switch, no stacking on an open position, liquidity, expiry-day and VIX-regime checks. |
| 7 | Paper trading | **Built.** Approved ideas become paper positions, marked on every run and settled at expiry. Every signal and decision is logged with the paper outcome of approved ideas (`fno-research review log`). Next: run it for 2–4 weeks. |
| 8 | Live with human review | Not started, deliberately. |

## How it works

```
 NSE website ─┐  option chain, spot, VIX, FII/DII
 Yahoo ───────┤  daily OHLCV                          ┌─ Technical group ─────────────┐
 RSS / Google ┤  headlines                            │ trend · momentum · volatility │
 (Kite later) ┘                                       │ volume                         │
        │                                             └───────────────┬───────────────┘
        ▼                                             ┌─ Context group ┴──────────────┐
  Feature store ◄──── every run ────────────────────  │ options positioning · flows   │
                                                      │ news (Claude)                 │
                                                      └───────────────┬───────────────┘
                                                                      ▼
                         Aggregator: weight × confidence per group; veto on conflict;
                         allocation multiplier (full size only on confluence)
                                                                      ▼
                         Strategy: bull call / bear put debit spread, short leg at OI wall
                                                                      ▼
                         Risk engine: size by budget × multiplier, then every check
                                                                      ▼
                         Review queue ──approve──► Paper book ──► kill switch
```

Each agent returns a score in [-1, 1] and a confidence in [0, 1]. An agent that fails
(for example because the site is down) returns zero confidence, and the run carries on.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard,dev]"
cp .env.example .env
```

No keys are needed for the free NSE/Yahoo data. `ANTHROPIC_API_KEY` turns on the news agent.

## Running

```bash
streamlit run dashboard/app.py                 # research, review queue, paper book, backtest
fno-research run NIFTY                         # one run (free data by default)
fno-research run BANKNIFTY --source sample     # synthetic data, works offline
fno-research review list                       # pending ideas
fno-research review log                        # every run, decision and paper outcome
fno-research review approve <id> --note "..."  # approve -> paper position
fno-research paper                             # positions, P&L, kill switch
fno-research paper reset-kill-switch
fno-research backtest NIFTY --years 3          # walk-forward test of technical agents
python scripts/scenario_session.py             # scripted multi-day session (see below)
pytest                                         # offline test suite
```

Run it on a schedule during market hours (for example every 15 minutes) to build the feature
store and mark paper positions. Keep the frequency modest; the NSE client caches and throttles.

## Backtesting

`fno-research backtest` walks forward day by day. Each technical agent sees only the bars up
to that close, and its score is compared with the index return over the next 1, 3 and 5
sessions. Read `hit_rate` against `base_up_rate`, and `edge_bps` against `market_bps` (what
simply holding earned). Positive edge in a rising market is not skill by itself.

Context agents need history that isn't freely downloadable (past option chains, news), so
`backtest.evaluate_stored` scores them from what the feature store records on live runs.

## Scenario session

`scripts/scenario_session.py` plays out a synthetic week with a scripted reviewer:

1. A bullish confluence idea is approved and paper-traded.
2. A technical-bearish vs FII-bullish conflict is vetoed.
3. A second idea on the same index is blocked while one is open.
4. A gap-down trips the daily-loss kill switch, which then blocks a new idea.
5. The position settles at expiry.

It exits non-zero if any expectation fails.

## Layout

```
src/fno_research/
  agents/     technical.py (trend, momentum, volatility, volume), options_positioning.py,
              flows.py, news.py (Claude)
  analytics/  indicators.py, options.py (Black-Scholes, IV, PCR, max pain), regime.py
  data/       nse.py + yahoo.py + free.py (free web data), news.py (RSS), kite.py, sample.py
  aggregator.py  strategy.py  risk.py  review.py  paper.py  store.py  backtest.py  pipeline.py
dashboard/app.py
scripts/scenario_session.py
```

## Notes and limitations

- **NSE endpoints are undocumented** (they're what nseindia.com's own pages call). They need
  browser-like headers and cookies, which the client handles, and NSE changes them without
  notice. The parsers accept both the older `option-chain-indices` and the newer
  `option-chain-v3` formats. Poll gently. NSE also blocks many cloud/datacenter IPs, so run
  it from your own machine.
- NSE reports option OI in contracts; the app multiplies by lot size so OI is in units, like
  Kite. Lot sizes are in `config.py` and can be overridden with `FNO_LOT_<INDEX>` when NSE
  revises them.
- Yahoo gives up to ~60 days of 15-minute history and years of daily data. Index volume on
  Yahoo can be patchy; the volume agent abstains when volume is missing.
- **OI build-up** reads each near-the-money strike's price change with its OI change (long build-up, short build-up, short covering, long unwinding). Put writing counts as bullish and call writing as bearish.
- **F&O ban list** comes from NSE's `fo_secban.csv`. Index options are never banned, so the check only matters once stock options are added; for a stock, a missing list blocks the trade.
- Margin without a broker is estimated as the spread's net debit. With Kite it comes from
  Kite's basket-margin API.
- Paper fills assume you pay the ask and receive the bid. Expiry settlement uses the spot
  seen at the first mark after 15:30 on expiry day, so run once after the close.
