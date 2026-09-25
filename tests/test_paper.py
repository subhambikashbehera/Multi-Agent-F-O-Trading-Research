from datetime import datetime

import pytest

from fno_research.data.sample import SampleDataProvider
from fno_research.models import (
    AggregateSignal,
    Direction,
    OptionChain,
    ResearchReport,
)
from fno_research.paper import PaperBook, settlement_value
from fno_research.strategy import build_idea


def make_report(chain, direction=Direction.BULLISH):
    view = AggregateSignal(score=0.5, confidence=0.6, direction=direction, agreement=1,
                           contributions={}, allocation_multiplier=1)
    idea = build_idea(chain, view)
    return ResearchReport(id=f"r{direction.value}", created_at=chain.as_of,
                          underlying=chain.underlying, spot=chain.spot, signals=[],
                          aggregate=view, idea=idea, risk=None, summary="")


def reprice(chain: OptionChain, bump: float, when: datetime) -> OptionChain:
    """Same chain with every CE bumped by `bump` and PE cut by `bump` (a rally)."""
    quotes = []
    for q in chain.quotes:
        d = bump if q.option_type == "CE" else -bump
        quotes.append(q.model_copy(update={
            "last_price": max(q.last_price + d, 0.05), "bid": max(q.bid + d, 0.05),
            "ask": max(q.ask + d, 0.1),
        }))
    return chain.model_copy(update={"quotes": quotes, "as_of": when})


@pytest.fixture
def book(tmp_path):
    return PaperBook(tmp_path / "paper.db")


def test_open_mark_and_daily_pnl(book, chain):
    report = make_report(chain)
    pid = book.open_from_report(report, when=chain.as_of)
    pos = book.positions("open")[0]
    assert pos["id"] == pid and pos["entry_cost"] == pytest.approx(report.idea.net_debit)

    later = reprice(chain, 10, chain.as_of.replace(hour=14))
    assert book.mark(later) == 1
    pos = book.positions("open")[0]
    # Long CE +10, short CE +10 -> spread value roughly unchanged; P&L is the bid/ask cost.
    assert pos["unrealized_pnl"] == pytest.approx(pos["last_value"] - pos["entry_cost"])
    assert book.daily_pnl(chain.as_of.date()) == pytest.approx(pos["unrealized_pnl"])


def test_daily_pnl_uses_previous_close_as_start(book, chain):
    book.open_from_report(make_report(chain), when=chain.as_of)
    day1 = chain.as_of.replace(hour=15, minute=25)
    book.mark(chain.model_copy(update={"as_of": day1}))
    v1 = book.positions("open")[0]["last_value"]
    day2 = datetime(2026, 9, 23, 11, 0)
    long_leg = book.positions("open")[0]["legs"][0]
    # Only the long leg's bid moves up by 20 on day 2.
    quotes = [q.model_copy(update={"bid": q.bid + 20})
              if (q.strike, q.option_type) == (long_leg.strike, "CE") else q
              for q in chain.quotes]
    book.mark(chain.model_copy(update={"as_of": day2, "quotes": quotes}))
    assert book.daily_pnl(day2.date()) == pytest.approx(20 * chain.lot_size)
    assert book.positions("open")[0]["last_value"] == pytest.approx(v1 + 20 * chain.lot_size)


def test_settles_at_expiry(book, chain):
    report = make_report(chain)
    book.open_from_report(report, when=chain.as_of)
    after = chain.expiry
    expired = chain.model_copy(update={
        "as_of": datetime(after.year, after.month, after.day, 15, 31), "spot": 26_000.0,
    })
    book.mark(expired)
    pos = book.positions("closed")[0]
    idea = report.idea
    assert pos["exit_value"] == pytest.approx(settlement_value(idea.legs, idea.lot_size, 26_000))
    # Far above both strikes: a bull call spread pays its full width.
    width = (idea.legs[1].strike - idea.legs[0].strike) * idea.lot_size
    assert pos["exit_value"] == pytest.approx(width)
    assert pos["realized_pnl"] == pytest.approx(width - idea.net_debit)


def test_kill_switch_trips_and_resets(book, chain):
    book.open_from_report(make_report(chain), when=chain.as_of)
    crash = reprice(chain, -150, chain.as_of.replace(hour=14))
    book.mark(crash)
    loss = -book.daily_pnl(chain.as_of.date())
    assert loss > 0
    assert not book.check_kill_switch(limit=loss + 1, day=chain.as_of.date())["active"]
    state = book.check_kill_switch(limit=loss - 1, day=chain.as_of.date())
    assert state["active"] and "loss limit" in state["reason"]
    assert book.kill_switch()["active"]  # persisted
    book.reset_kill_switch()
    assert not book.kill_switch()["active"]


def test_manual_close(book, chain):
    pid = book.open_from_report(make_report(chain), when=chain.as_of)
    book.close(pid)
    assert book.positions("closed")[0]["realized_pnl"] == pytest.approx(0)
    with pytest.raises(ValueError):
        book.close(pid)


def test_other_underlying_is_not_marked(book, chain):
    book.open_from_report(make_report(chain), when=chain.as_of)
    bank = SampleDataProvider().option_chain("BANKNIFTY")
    assert book.mark(bank) == 0


def test_report_without_idea_cannot_open(book, chain):
    report = make_report(chain, Direction.NEUTRAL)
    with pytest.raises(ValueError):
        book.open_from_report(report)
