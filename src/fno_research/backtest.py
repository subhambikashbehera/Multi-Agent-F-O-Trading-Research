"""Standalone backtests: does each agent's score predict forward index returns?

Technical agents are walk-forward tested on daily history: on each day they see only the
bars up to that close, and the score is compared with the return over the next N sessions.
Context agents (options positioning, flows, news) need history nobody gives away free, so
`evaluate_stored` scores them from what the feature store has recorded on live runs.

These measure signal quality, not strategy P&L (no option prices, costs or slippage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fno_research.agents.technical import TECHNICAL_AGENTS, TechnicalAgent
from fno_research.aggregator import aggregate
from fno_research.config import AgentWeights
from fno_research.store import FeatureStore

HORIZONS = (1, 3, 5)
WINDOW = 300  # bars each agent sees per step; enough for every indicator to settle


@dataclass
class BacktestResult:
    signals: pd.DataFrame  # index = date, columns = <agent>_score, <agent>_conf, fwd_<n>
    summary: pd.DataFrame  # one row per (agent, horizon)


def forward_returns(close: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    return pd.DataFrame({f"fwd_{h}": close.shift(-h) / close - 1 for h in horizons})


def score_series(scores: pd.Series, confidence: pd.Series, fwd: pd.Series,
                 threshold: float) -> dict:
    df = pd.DataFrame({"s": scores, "c": confidence, "r": fwd}).dropna()
    active = df[df.s.abs() >= threshold]
    hits = np.sign(active.s) == np.sign(active.r)
    edge = np.sign(active.s) * active.r
    return {
        "days": len(df),
        "signals": len(active),
        "coverage": round(len(active) / len(df), 3) if len(df) else 0.0,
        "hit_rate": round(hits.mean(), 3) if len(active) else np.nan,
        "conf_weighted_hit": (round(float(np.average(hits, weights=active.c)), 3)
                              if len(active) and active.c.sum() > 0 else np.nan),
        "edge_bps": round(edge.mean() * 1e4, 1) if len(active) else np.nan,
        "long_bps": round(active[active.s > 0].r.mean() * 1e4, 1)
        if (active.s > 0).any() else np.nan,
        "short_bps": round(active[active.s < 0].r.mean() * 1e4, 1)
        if (active.s < 0).any() else np.nan,
        # Spearman rank correlation (information coefficient), computed without scipy.
        "ic": round(df.s.rank().corr(df.r.rank()), 3) if len(df) > 2 else np.nan,
        "base_up_rate": round((df.r > 0).mean(), 3) if len(df) else np.nan,
        # Average forward return of simply holding. A mostly-long agent in a rising market
        # shows positive edge_bps without skill; judge it against this.
        "market_bps": round(df.r.mean() * 1e4, 1) if len(df) else np.nan,
    }


def backtest_technical(frame: pd.DataFrame, agents: list[TechnicalAgent] | None = None,
                       warmup: int = 120, threshold: float = 0.15,
                       weights: AgentWeights | None = None) -> BacktestResult:
    agents = agents or [cls() for cls in TECHNICAL_AGENTS]
    weights = weights or AgentWeights()
    rows = []
    for t in range(warmup, len(frame)):
        window = frame.iloc[max(0, t + 1 - WINDOW) : t + 1]
        sigs = [a.analyse_frame(window) for a in agents]
        row = {"date": frame.index[t]}
        for s in sigs:
            row[f"{s.agent}_score"] = s.score
            row[f"{s.agent}_conf"] = s.confidence
        group = aggregate(sigs, weights).groups.get("technical")
        row["technical_group_score"] = group.score if group else 0.0
        row["technical_group_conf"] = group.confidence if group else 0.0
        rows.append(row)
    signals = pd.DataFrame(rows).set_index("date")
    signals = signals.join(forward_returns(frame["close"]))

    names = [a.name for a in agents] + ["technical_group"]
    summary = []
    for name in names:
        for h in HORIZONS:
            summary.append({
                "agent": name, "horizon": h,
                **score_series(signals[f"{name}_score"], signals[f"{name}_conf"],
                               signals[f"fwd_{h}"], threshold),
            })
    return BacktestResult(signals=signals, summary=pd.DataFrame(summary))


def evaluate_stored(store: FeatureStore, underlying: str, agent: str, daily: pd.DataFrame,
                    threshold: float = 0.15) -> pd.DataFrame:
    """Score a context agent from the scores recorded on live runs (last run of each day)."""
    scores = store.feature_history(underlying, agent, "score")
    conf = store.feature_history(underlying, agent, "confidence")
    if scores.empty:
        return pd.DataFrame()
    per_day = pd.DataFrame({
        "s": scores.set_index("ts")["value"], "c": conf.set_index("ts")["value"],
    })
    per_day = per_day.groupby(per_day.index.normalize()).last()
    close = daily["close"].copy()
    close.index = pd.DatetimeIndex(close.index).normalize()
    fwd = forward_returns(close)
    joined = per_day.join(fwd, how="inner")
    return pd.DataFrame([
        {"agent": agent, "horizon": h,
         **score_series(joined.s, joined.c, joined[f"fwd_{h}"], threshold)}
        for h in HORIZONS
    ])
