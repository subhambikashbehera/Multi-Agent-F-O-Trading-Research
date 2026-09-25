"""Human review queue. Reports that pass the risk engine wait here for a person to decide.

Nothing in this project places orders. Approving an idea records the decision and, in
the dashboard and CLI, opens a paper position.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from fno_research.models import ResearchReport

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
BLOCKED = "blocked_by_risk"
NO_TRADE = "no_trade"


class ReviewQueue:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                underlying TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                reviewer_note TEXT,
                decided_at TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, report: ResearchReport) -> str:
        if report.idea is None:
            status = NO_TRADE
        elif report.risk is not None and report.risk.approved:
            status = PENDING
        else:
            status = BLOCKED
        self.conn.execute(
            "INSERT INTO reports (id, created_at, underlying, status, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (report.id, report.created_at.isoformat(), report.underlying, status,
             report.model_dump_json()),
        )
        self.conn.commit()
        return status

    def list(self, status: str | None = None, limit: int = 50) -> list[dict]:
        query = "SELECT id, created_at, underlying, status, payload, reviewer_note, decided_at " \
                "FROM reports"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC LIMIT ?"
        rows = self.conn.execute(query, (*params, limit)).fetchall()
        return [
            {
                "id": r[0], "created_at": r[1], "underlying": r[2], "status": r[3],
                "report": ResearchReport.model_validate_json(r[4]),
                "reviewer_note": r[5], "decided_at": r[6],
            }
            for r in rows
        ]

    def decide(self, report_id: str, approve: bool, note: str = "") -> ResearchReport:
        """Record the decision and return the report (approve -> open a paper position)."""
        cur = self.conn.execute(
            "UPDATE reports SET status = ?, reviewer_note = ?, decided_at = ? "
            "WHERE id = ? AND status = ?",
            (APPROVED if approve else REJECTED, note, datetime.now().isoformat(),
             report_id, PENDING),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            raise ValueError(f"Report {report_id} is not pending review")
        row = self.conn.execute("SELECT payload FROM reports WHERE id = ?",
                                (report_id,)).fetchone()
        return ResearchReport.model_validate_json(row[0])


def decision_log(queue: ReviewQueue, book, limit: int = 200) -> list[dict]:
    """Every run with its decision and, for approved ideas, the paper outcome so far."""
    positions = {p["report_id"]: p for p in book.positions()}
    log = []
    for row in queue.list(limit=limit):
        report: ResearchReport = row["report"]
        pos = positions.get(row["id"])
        if pos is None:
            outcome, pnl = "", None
        elif pos["status"] == "closed":
            outcome, pnl = "closed", pos["realized_pnl"]
        else:
            outcome, pnl = "open", pos["unrealized_pnl"]
        log.append({
            "time": row["created_at"][:16],
            "id": row["id"],
            "underlying": row["underlying"],
            "status": row["status"],
            "direction": report.aggregate.direction.value,
            "score": report.aggregate.score,
            "allocation": report.aggregate.allocation_multiplier,
            "veto": report.aggregate.veto,
            "strategy": report.idea.strategy if report.idea else "",
            "note": row["reviewer_note"] or "",
            "paper": outcome,
            "pnl": round(pnl, 2) if pnl is not None else None,
        })
    return log
