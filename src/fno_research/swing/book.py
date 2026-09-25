"""Swing review queue and paper book.

Ideas that pass risk wait for review. An approved idea opens a paper position at its
planned entry. Each scan walks the position forward bar by bar through the new daily
bars: stop first (a gap below the stop fills at the open), then target, then the time
stop. When one bar touches both stop and target we assume the stop, the conservative
reading of daily data.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from fno_research.swing.models import SHORT, SwingIdea

SCHEMA = """
CREATE TABLE IF NOT EXISTS swing_ideas (
    id TEXT PRIMARY KEY, created_at TEXT, symbol TEXT, status TEXT, payload TEXT,
    note TEXT, decided_at TEXT
);
CREATE TABLE IF NOT EXISTS swing_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, idea_id TEXT UNIQUE, symbol TEXT, name TEXT,
    industry TEXT, qty INTEGER, entry REAL, stop REAL, target REAL, opened_on TEXT,
    time_stop_sessions INTEGER, sessions_held INTEGER DEFAULT 0, last_date TEXT,
    last_close REAL, status TEXT, exit_price REAL, exit_on TEXT, exit_reason TEXT, pnl REAL
);
CREATE TABLE IF NOT EXISTS swing_daily_pnl (date TEXT PRIMARY KEY, pnl REAL);
"""
PENDING, APPROVED, REJECTED, BLOCKED = "pending", "approved", "rejected", "blocked_by_risk"


class SwingBook:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # -- review queue ---------------------------------------------------------

    def add_ideas(self, ideas: list[SwingIdea]) -> None:
        for idea in ideas:
            self.conn.execute(
                "INSERT OR IGNORE INTO swing_ideas (id, created_at, symbol, status, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (idea.id, idea.created_at.isoformat(), idea.symbol,
                 PENDING if idea.risk.approved else BLOCKED, idea.model_dump_json()),
            )
        self.conn.commit()

    def ideas(self, status: str | None = None, limit: int = 200) -> list[dict]:
        query, params = "SELECT * FROM swing_ideas", ()
        if status:
            query, params = query + " WHERE status = ?", (status,)
        rows = self.conn.execute(query + " ORDER BY created_at DESC LIMIT ?",
                                 (*params, limit)).fetchall()
        return [dict(r) | {"idea": SwingIdea.model_validate_json(r["payload"])} for r in rows]

    def decide(self, idea_id: str, approve: bool, note: str = "", paper=None) -> int | None:
        """Record the decision; approving opens a paper position and returns its id.
        Long ideas go into this book; bearish option spreads go into `paper` (PaperBook)."""
        cur = self.conn.execute(
            "UPDATE swing_ideas SET status = ?, note = ?, decided_at = ? "
            "WHERE id = ? AND status = ?",
            (APPROVED if approve else REJECTED, note, datetime.now().isoformat(), idea_id,
             PENDING),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Swing idea {idea_id} is not pending review")
        self.conn.commit()
        if not approve:
            return None
        idea = SwingIdea.model_validate_json(self.conn.execute(
            "SELECT payload FROM swing_ideas WHERE id = ?", (idea_id,)).fetchone()[0])
        if idea.side == SHORT:
            if paper is None:
                raise ValueError("A PaperBook is needed to open a bearish option spread")
            return paper.open_idea(idea.option_idea, idea.id)
        p = idea.plan
        cur = self.conn.execute(
            "INSERT INTO swing_positions (idea_id, symbol, name, industry, qty, entry, stop, "
            "target, opened_on, time_stop_sessions, last_date, last_close, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')",
            (idea.id, p.symbol, idea.stock.name, idea.stock.industry, p.qty, p.entry, p.stop,
             p.target, idea.as_of.isoformat(), p.time_stop_sessions, idea.as_of.isoformat(),
             p.entry),
        )
        self.conn.commit()
        return cur.lastrowid

    # -- paper book -------------------------------------------------------------

    def positions(self, status: str | None = None) -> list[dict]:
        query, params = "SELECT * FROM swing_positions", ()
        if status:
            query, params = query + " WHERE status = ?", (status,)
        rows = self.conn.execute(query + " ORDER BY opened_on DESC, id DESC", params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["unrealized_pnl"] = ((d["last_close"] - d["entry"]) * d["qty"]
                                   if d["status"] == "open" else 0.0)
            out.append(d)
        return out

    def _add_daily(self, day: str, pnl: float) -> None:
        self.conn.execute(
            "INSERT INTO swing_daily_pnl VALUES (?, ?) "
            "ON CONFLICT(date) DO UPDATE SET pnl = pnl + excluded.pnl", (day, pnl))

    def update(self, frames: dict[str, pd.DataFrame]) -> int:
        """Walk open positions through bars newer than the last one processed."""
        exits = 0
        for pos in self.positions("open"):
            frame = frames.get(pos["symbol"])
            if frame is None:
                continue
            new = frame[frame.index.normalize() > pd.Timestamp(pos["last_date"])]
            prev_close, held = pos["last_close"], pos["sessions_held"]
            for ts, bar in new.iterrows():
                day = ts.date().isoformat()
                held += 1
                exit_price = reason = None
                if bar.open <= pos["stop"]:
                    exit_price, reason = bar.open, "gap below stop"
                elif bar.low <= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop"
                elif bar.open >= pos["target"]:
                    exit_price, reason = bar.open, "gap above target"
                elif bar.high >= pos["target"]:
                    exit_price, reason = pos["target"], "target"
                elif held >= pos["time_stop_sessions"]:
                    exit_price, reason = bar.close, "time stop"
                mark = exit_price if exit_price is not None else bar.close
                self._add_daily(day, (mark - prev_close) * pos["qty"])
                prev_close = mark
                if exit_price is not None:
                    self.conn.execute(
                        "UPDATE swing_positions SET status = 'closed', exit_price = ?, "
                        "exit_on = ?, exit_reason = ?, pnl = ?, sessions_held = ?, "
                        "last_date = ?, last_close = ? WHERE id = ?",
                        (round(exit_price, 2), day, reason,
                         round((exit_price - pos["entry"]) * pos["qty"], 2), held, day,
                         exit_price, pos["id"]),
                    )
                    exits += 1
                    break
            else:
                if len(new):
                    self.conn.execute(
                        "UPDATE swing_positions SET sessions_held = ?, last_date = ?, "
                        "last_close = ? WHERE id = ?",
                        (held, new.index[-1].date().isoformat(), float(new["close"].iloc[-1]),
                         pos["id"]),
                    )
        self.conn.commit()
        return exits

    def close(self, position_id: int, price: float | None = None) -> None:
        pos = next((p for p in self.positions("open") if p["id"] == position_id), None)
        if pos is None:
            raise ValueError(f"Swing position {position_id} is not open")
        price = price if price is not None else pos["last_close"]
        today = date.today().isoformat()
        self._add_daily(today, (price - pos["last_close"]) * pos["qty"])
        self.conn.execute(
            "UPDATE swing_positions SET status = 'closed', exit_price = ?, exit_on = ?, "
            "exit_reason = 'manual', pnl = ? WHERE id = ?",
            (price, today, round((price - pos["entry"]) * pos["qty"], 2), position_id))
        self.conn.commit()

    def daily_pnl(self, day: date) -> float:
        row = self.conn.execute("SELECT pnl FROM swing_daily_pnl WHERE date = ?",
                                (day.isoformat(),)).fetchone()
        return row[0] if row else 0.0
