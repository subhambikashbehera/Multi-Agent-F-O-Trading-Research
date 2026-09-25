"""Feature store: every run's agent features, market context, FII/DII flows and a full
option-chain snapshot, so agents can be calibrated and backtested later from real data."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from fno_research.models import FlowSnapshot, OptionChain, ResearchReport

SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    report_id TEXT, ts TEXT, underlying TEXT, agent TEXT, name TEXT, value REAL, text TEXT
);
CREATE INDEX IF NOT EXISTS features_lookup ON features (underlying, agent, name, ts);
CREATE TABLE IF NOT EXISTS context (
    report_id TEXT PRIMARY KEY, ts TEXT, underlying TEXT, data_source TEXT, spot REAL,
    vix REAL, vix_percentile REAL, vol_regime TEXT, days_to_expiry INTEGER, expiry_tag TEXT,
    score REAL, confidence REAL, direction TEXT, allocation_multiplier REAL, veto INTEGER
);
CREATE TABLE IF NOT EXISTS flows (date TEXT PRIMARY KEY, fii_net REAL, dii_net REAL);
CREATE TABLE IF NOT EXISTS chains (
    ts TEXT, underlying TEXT, expiry TEXT, spot REAL, data_source TEXT, payload TEXT
);
"""


class FeatureStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def record(self, report: ResearchReport, chain: OptionChain | None) -> None:
        ts = report.created_at.isoformat()
        rows = []
        for s in report.signals:
            rows.append((report.id, ts, report.underlying, s.agent, "score", s.score, None))
            rows.append((report.id, ts, report.underlying, s.agent, "confidence",
                         s.confidence, None))
            for name, value in s.features.items():
                if isinstance(value, (int, float)):
                    rows.append((report.id, ts, report.underlying, s.agent, name, value, None))
                elif value is not None:
                    rows.append((report.id, ts, report.underlying, s.agent, name, None,
                                 str(value)))
        self.conn.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        c, a = report.context, report.aggregate
        self.conn.execute(
            "INSERT OR REPLACE INTO context VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (report.id, ts, report.underlying, c.data_source if c else None, report.spot,
             c.vix if c else None, c.vix_percentile if c else None,
             c.vol_regime if c else None, c.days_to_expiry if c else None,
             c.expiry_tag if c else None, a.score, a.confidence, a.direction.value,
             a.allocation_multiplier, int(a.veto)),
        )
        if chain is not None:
            self.conn.execute(
                "INSERT INTO chains VALUES (?, ?, ?, ?, ?, ?)",
                (chain.as_of.isoformat(), chain.underlying, chain.expiry.isoformat(),
                 chain.spot, c.data_source if c else None, chain.model_dump_json()),
            )
        self.conn.commit()

    def save_flows(self, snapshot: FlowSnapshot) -> None:
        self.conn.execute("INSERT OR REPLACE INTO flows VALUES (?, ?, ?)",
                          (snapshot.date.isoformat(), snapshot.fii_net, snapshot.dii_net))
        self.conn.commit()

    def flows(self, days: int = 5) -> list[FlowSnapshot]:
        rows = self.conn.execute(
            "SELECT date, fii_net, dii_net FROM flows ORDER BY date DESC LIMIT ?", (days,)
        ).fetchall()
        return [FlowSnapshot(date=date.fromisoformat(d), fii_net=f, dii_net=di)
                for d, f, di in reversed(rows)]

    def feature_history(self, underlying: str, agent: str, name: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT ts, value, text FROM features WHERE underlying = ? AND agent = ? "
            "AND name = ? ORDER BY ts",
            self.conn, params=(underlying, agent, name), parse_dates=["ts"],
        )

    def chains(self, underlying: str) -> list[OptionChain]:
        rows = self.conn.execute(
            "SELECT payload FROM chains WHERE underlying = ? ORDER BY ts", (underlying,)
        ).fetchall()
        return [OptionChain.model_validate_json(r[0]) for r in rows]
