# Multi-Agent F&O Trading Research

A multi-agent research pipeline for Indian index futures and options. Specialised agents
read price action, options positioning and news, a weighted aggregator combines them, and a
risk engine plus human review sit in front of any order.

**This is a research tool. It never places orders.** Approving an idea in the review queue
only records your decision.

## How it works

```
             ┌──────────────────┐
 Kite ──────►│ Price action     │  EMA20/50 trend, ATR distance, RSI, Supertrend (daily + 15m)
 Connect     ├──────────────────┤
     └──────►│ Options          │  PCR, fresh OI flow, put/call OI walls, max pain, ATM IV, skew
             │ positioning      │
 RSS news ──►├──────────────────┤
             │ News (Claude)    │  Structured sentiment + key drivers + event risk
             └────────┬─────────┘
                      ▼  score ∈ [-1, 1], confidence ∈ [0, 1] from each agent
             ┌──────────────────┐
             │ Aggregator       │  weight × confidence; agreement across agents
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ Strategy builder │  Bull call / bear put debit spread, short leg at the OI wall
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ Risk engine      │  sizes to the risk budget, then every check must pass
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ Review queue     │  a person approves or rejects (SQLite)
             └──────────────────┘
```

- **Hybrid agents.** The price-action and options agents are deterministic Python, so their
  signals are testable. Only the news agent calls Claude (`claude-opus-5`, structured output
  via `messages.parse`). If `ANTHROPIC_API_KEY` is not set, the news agent is skipped and has no
  vote. If an agent fails at runtime, it returns zero confidence and the run carries on.
- **Aggregator.** Each agent's weight is multiplied by its confidence. `agreement` is 1 when
  all confident agents point the same way and 0 when they cancel out. The final confidence is
  scaled down when agents disagree.
- **Risk engine checks.** Minimum confidence and agreement, defined risk only, max loss
  within `FNO_MAX_RISK_PCT` of `FNO_CAPITAL`, a lot cap, no new positions on expiry day, and
  minimum OI plus a maximum bid/ask spread on every leg. Any failed check blocks the idea.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard,dev]"
cp .env.example .env   # fill in Kite and Anthropic keys
```

### Kite Connect login (daily)

Kite access tokens expire every morning. Get today's token from either:

- the dashboard sidebar (log in, then paste the `request_token`), or
- `fno-research kite-login`, and put the printed `KITE_ACCESS_TOKEN` in `.env`.

You need a Kite Connect app with API key and secret from https://developers.kite.trade.

## Running

```bash
streamlit run dashboard/app.py          # dashboard: research + review queue
fno-research run NIFTY                  # one headless run with live Kite data
fno-research run BANKNIFTY --sample     # synthetic data, no credentials needed
pytest                                  # tests run fully offline
```

Supported underlyings: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY. You can adjust agent weights and
risk limits in the dashboard sidebar for each run.

## Layout

```
src/fno_research/
  agents/          price_action.py, options_positioning.py, news.py (Claude)
  analytics/       indicators.py (EMA, RSI, ATR, Supertrend), options.py (BS, IV, PCR, max pain)
  data/            kite.py (live), news.py (RSS), sample.py (synthetic, for tests/offline)
  aggregator.py    weighted, confidence-aware vote
  strategy.py      direction → defined-risk spread
  risk.py          sizing + risk checks
  review.py        SQLite approval queue
  pipeline.py      wires it together
dashboard/app.py   Streamlit UI
```

## Notes and limitations

- Kite's quote API does not return the previous day's OI. OI change is measured against the
  snapshot this app saved on an earlier trading day (`data/cache/`), so it reads zero on the
  first day you run it.
- IV is backed out from last traded prices with Black-Scholes (r = 6.5%). This is noisy for
  illiquid strikes.
- Sample mode is synthetic. The dashboard labels those runs, so don't read anything into their
  signals.

## Roadmap

- Bull/bear researcher debate before the aggregator, as in
  [TradingAgents](https://github.com/TauricResearch/TradingAgents).
- Backtest harness that replays stored chains and scores agent calibration.
- IV-regime aware structures (credit spreads or iron condors when IV is rich and the view is
  neutral).
- FII/DII participant-wise OI and India VIX as extra inputs.
- Paper-trading ledger for approved ideas.
