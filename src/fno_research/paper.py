"""Paper trading book and the daily-loss kill switch.

Approved ideas open a paper position at the prices the idea was built with. Each run marks
open positions to market from the fresh option chain (closing prices: bid for longs, ask for
shorts) and settles them at intrinsic value once expiry has passed. If the day's P&L falls
below the daily loss limit the kill switch trips and the risk engine blocks every new idea
until someone resets it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

from fno_research.models import OptionChain, OptionLeg, ResearchReport, TradeIdea

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, report_id TEXT UNIQUE, opened_at TEXT,
    underlying TEXT, expiry TEXT, strategy TEXT, lot_size INTEGER, legs TEXT,
    entry_cost REAL, status TEXT, closed_at TEXT, exit_value REAL, realized_pnl REAL,
    last_value REAL, last_marked_at TEXT
);
CREATE TABLE IF NOT EXISTS marks (position_id INTEGER, ts TEXT, date TEXT, value REAL);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


def _intrinsic(leg: OptionLeg, spot: float) -> float:
    return max(spot - leg.strike, 0) if leg.option_type == "CE" else max(leg.strike - spot, 0)


def position_value(legs: list[OptionLeg], lot_size: int, chain: OptionChain) -> float | None:
    """What closing the position now would realise, or None if a leg has no quote."""
    total = 0.0
    for leg in legs:
        q = chain.get(leg.strike, leg.option_type)
        if q is None:
            return None
        if leg.action == "BUY":
            price = q.bid if q.bid > 0 else q.last_price
            total += price * leg.lots * lot_size
        else:
            price = q.ask if q.ask > 0 else q.last_price
            total -= price * leg.lots * lot_size
    return total


def settlement_value(legs: list[OptionLeg], lot_size: int, spot: float) -> float:
    return sum((1 if leg.action == "BUY" else -1) * _intrinsic(leg, spot) * leg.lots * lot_size
               for leg in legs)


class PaperBook:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # -- positions ---------------------------------------------------------

    def open_from_report(self, report: ResearchReport, when: datetime | None = None) -> int:
        if report.idea is None:
            raise ValueError("Report has no trade idea")
        return self.open_idea(report.idea, report.id, when)

    def open_idea(self, idea: TradeIdea, ref_id: str, when: datetime | None = None) -> int:
        """Open a paper position for any option structure; ref_id links it to its source."""
        when = when or datetime.now()
        cur = self.conn.execute(
            "INSERT INTO positions (report_id, opened_at, underlying, expiry, strategy, "
            "lot_size, legs, entry_cost, status, last_value, last_marked_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)",
            (ref_id, when.isoformat(), idea.underlying, idea.expiry.isoformat(),
             idea.strategy, idea.lot_size,
             json.dumps([leg.model_dump() for leg in idea.legs]), idea.net_debit,
             idea.net_debit, when.isoformat()),
        )
        self.conn.commit()
        return cur.lastrowid

    def positions(self, status: str | None = None) -> list[dict]:
        query, params = "SELECT * FROM positions", ()
        if status:
            query, params = query + " WHERE status = ?", (status,)
        rows = self.conn.execute(query + " ORDER BY opened_at DESC", params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["legs"] = [OptionLeg(**leg) for leg in json.loads(d["legs"])]
            d["expiry"] = date.fromisoformat(d["expiry"])
            d["unrealized_pnl"] = (d["last_value"] - d["entry_cost"]
                                   if d["status"] == "open" else 0.0)
            out.append(d)
        return out

    def _record_mark(self, pos_id: int, value: float, when: datetime) -> None:
        self.conn.execute("INSERT INTO marks VALUES (?, ?, ?, ?)",
                          (pos_id, when.isoformat(), when.date().isoformat(), value))
        self.conn.execute("UPDATE positions SET last_value = ?, last_marked_at = ? WHERE id = ?",
                          (value, when.isoformat(), pos_id))

    def _close(self, pos: dict, value: float, when: datetime) -> None:
        self._record_mark(pos["id"], value, when)
        self.conn.execute(
            "UPDATE positions SET status = 'closed', closed_at = ?, exit_value = ?, "
            "realized_pnl = ? WHERE id = ?",
            (when.isoformat(), value, value - pos["entry_cost"], pos["id"]),
        )

    def mark(self, chain: OptionChain) -> int:
        """Mark (or settle) every open position on this chain's underlying. Returns count."""
        when = chain.as_of
        marked = 0
        for pos in self.positions("open"):
            if pos["underlying"] != chain.underlying:
                continue
            expired = when.date() > pos["expiry"] or (
                when.date() == pos["expiry"] and (when.hour, when.minute) >= (15, 30)
            )
            if expired:
                # Settles at the spot we see now; accurate only if marked at expiry close.
                self._close(pos, settlement_value(pos["legs"], pos["lot_size"], chain.spot),
                            when)
                marked += 1
            elif chain.expiry == pos["expiry"]:
                value = position_value(pos["legs"], pos["lot_size"], chain)
                if value is not None:
                    self._record_mark(pos["id"], value, when)
                    marked += 1
        self.conn.commit()
        return marked

    def close(self, position_id: int, when: datetime | None = None) -> None:
        """Close at the last marked value."""
        pos = next((p for p in self.positions("open") if p["id"] == position_id), None)
        if pos is None:
            raise ValueError(f"Position {position_id} is not open")
        self._close(pos, pos["last_value"], when or datetime.now())
        self.conn.commit()

    def daily_pnl(self, day: date) -> float:
        """P&L for `day`: current (or exit) value minus the value at the start of the day."""
        total = 0.0
        for pos in self.positions():
            opened = datetime.fromisoformat(pos["opened_at"]).date()
            closed = datetime.fromisoformat(pos["closed_at"]).date() if pos["closed_at"] \
                else None
            if opened > day or (closed and closed < day):
                continue
            prior = self.conn.execute(
                "SELECT value FROM marks WHERE position_id = ? AND date < ? "
                "ORDER BY ts DESC LIMIT 1", (pos["id"], day.isoformat())
            ).fetchone()
            start = pos["entry_cost"] if opened == day or prior is None else prior[0]
            end = pos["exit_value"] if closed == day else pos["last_value"]
            total += end - start
        return total

    # -- kill switch ------------------------------------------------------

    def kill_switch(self) -> dict:
        row = self.conn.execute("SELECT value FROM state WHERE key = 'kill_switch'").fetchone()
        return json.loads(row[0]) if row else {"active": False}

    def check_kill_switch(self, limit: float, day: date, extra_pnl: float = 0.0) -> dict:
        """Trip when today's P&L (options book plus `extra_pnl`, e.g. the swing cash book)
        breaches the limit."""
        state = self.kill_switch()
        if state.get("active"):
            return state
        pnl = self.daily_pnl(day) + extra_pnl
        if pnl <= -limit:
            state = {
                "active": True,
                "tripped_at": datetime.now().isoformat(),
                "reason": f"Daily P&L ₹{pnl:,.0f} breached the ₹{limit:,.0f} loss limit.",
            }
            self._set_state(state)
        return state

    def reset_kill_switch(self) -> None:
        self._set_state({"active": False})

    def _set_state(self, state: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO state VALUES ('kill_switch', ?)",
                          (json.dumps(state),))
        self.conn.commit()
