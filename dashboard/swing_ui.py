"""Swing trading desk for the Streamlit dashboard."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fno_research.analytics.indicators import ema
from fno_research.config import NSE_INDEX_NAMES, Settings
from fno_research.factory import build_swing
from fno_research.paper import PaperBook
from fno_research.swing.book import PENDING, SwingBook
from fno_research.swing.models import LONG, SHORT, ScanReport, SwingIdea

UNIVERSES = {
    "NIFTY LargeMidcap 250": "niftylargemidcap250",
    "NIFTY 200": "nifty200",
    "NIFTY 500": "nifty500",
    "NIFTY Midcap 150": "niftymidcap150",
    "NIFTY 100": "nifty100",
}
SIDE_COLOR = {LONG: "green", SHORT: "red"}


def swing_sidebar(settings: Settings) -> str:
    st.sidebar.title("Swing Desk")
    source = st.sidebar.radio(
        "Market data", ["NSE + Yahoo (free)", "Sample (synthetic)"],
        index=1 if settings.data_source == "sample" else 0, key="swing_source",
    )
    source = "sample" if source.startswith("Sample") else "nse"
    cfg = settings.swing
    if source == "nse":
        label = st.sidebar.selectbox("Universe", list(UNIVERSES), key="swing_universe")
        cfg.universe = UNIVERSES[label]
    else:
        st.sidebar.caption("Universe: 40 synthetic DEMO stocks")
    with st.sidebar.expander("Rules"):
        cfg.allow_short = st.checkbox("Bearish ideas (bear put spreads)", cfg.allow_short)
        cfg.capital = st.number_input("Swing capital (₹)", 50_000.0, 1e9, float(cfg.capital),
                                      50_000.0)
        cfg.risk_per_trade_pct = st.number_input("Risk per trade (%)", 0.1, 5.0,
                                                 float(cfg.risk_per_trade_pct), 0.1)
        cfg.max_open = int(st.number_input("Max open positions", 1, 30, cfg.max_open))
        cfg.max_per_industry = int(st.number_input("Max per industry", 1, 10,
                                                   cfg.max_per_industry))
        cfg.min_score = st.slider("Min |score|", 0.1, 0.9, cfg.min_score, 0.05)
        cfg.reward_risk = st.slider("Target (R multiple)", 1.0, 4.0, cfg.reward_risk, 0.5)
        cfg.time_stop_sessions = int(st.number_input("Time stop (sessions)", 3, 60,
                                                     cfg.time_stop_sessions))
        cfg.min_avg_value_cr = st.number_input("Min avg traded value (₹ cr)", 0.0, 500.0,
                                               float(cfg.min_avg_value_cr), 5.0)
    if not settings.has_anthropic:
        st.sidebar.info("ANTHROPIC_API_KEY not set: stock news isn't read.")
    return source


@st.cache_resource
def _books(path: str) -> tuple[SwingBook, PaperBook]:
    from pathlib import Path

    return SwingBook(Path(path)), PaperBook(Path(path))


def price_chart(frame: pd.DataFrame, idea: SwingIdea | None) -> go.Figure:
    f = frame.iloc[-130:]
    fig = go.Figure(go.Candlestick(x=f.index, open=f.open, high=f.high, low=f.low,
                                   close=f.close, name="Price"))
    for span, color in ((20, "#1f77b4"), (50, "#ff7f0e"), (200, "#7f7f7f")):
        fig.add_scatter(x=f.index, y=ema(frame["close"], span).iloc[-130:], name=f"EMA{span}",
                        line=dict(width=1.2, color=color))
    if idea and idea.plan:
        for level, name, color in ((idea.plan.entry, "Entry", "#444"),
                                   (idea.plan.stop, "Stop", "#c0392b"),
                                   (idea.plan.target, "Target", "#1e8449")):
            fig.add_hline(y=level, line_dash="dash", line_color=color,
                          annotation_text=f"{name} {level:,.2f}")
    if idea and idea.option_idea:
        for leg in idea.option_idea.legs:
            fig.add_hline(y=leg.strike, line_dash="dot",
                          annotation_text=f"{leg.action} {leg.strike:g} PE")
    fig.update_layout(height=420, margin=dict(t=20, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h"))
    return fig


def idea_card(idea: SwingIdea) -> None:
    s = idea.stock
    ok = idea.risk.approved
    with st.container(border=True):
        st.markdown(f"**{s.symbol}** · {s.name} · :{SIDE_COLOR[idea.side]}[{idea.side.upper()}] "
                    f"· score {s.score:+.2f} · {'✅ passes risk' if ok else '❌ blocked'}")
        st.caption(f"{s.industry} · RS vs NIFTY 60d {s.rs_60:+.1%} · avg value "
                   f"₹{s.avg_value_cr:,.0f} cr")
        if idea.plan:
            p = idea.plan
            st.markdown(
                f"Buy **{p.qty}** @ {p.entry:,.2f} · stop {p.stop:,.2f} ({p.stop_pct:.1f}%) · "
                f"target {p.target:,.2f} ({p.reward_risk:.1f}R) · risk ₹{p.risk_amount:,.0f} · "
                f"exit after {p.time_stop_sessions} sessions if neither hits")
        elif idea.option_idea:
            o = idea.option_idea
            legs = " / ".join(f"{leg.action} {leg.lots}×{o.lot_size} {leg.strike:g}PE @ "
                              f"{leg.price}" for leg in o.legs)
            st.markdown(f"{o.strategy}, expiry {o.expiry:%d %b}: {legs} · max loss "
                        f"₹{o.max_loss:,.0f} · max profit ₹{o.max_profit:,.0f}")
        if not ok:
            st.caption("Blocked by: " + "; ".join(f"{c.name} ({c.detail})"
                                                  for c in idea.risk.failures))


def scanner_tab(settings: Settings, source: str, book: SwingBook, paper: PaperBook) -> None:
    if st.button("Scan universe", type="primary"):
        bar = st.progress(0.0, text="Loading universe…")
        try:
            scanner = build_swing(settings, source)

            def progress(i, n, sym):
                bar.progress((i + 1) / n, text=f"Scoring {sym} ({i + 1}/{n})")

            report = scanner.run(progress)
        except Exception as exc:
            bar.empty()
            st.error(f"Scan failed: {exc}")
            return
        bar.empty()
        book.add_ideas(report.ideas)
        st.session_state["swing_scan"] = (report, scanner.data)
        st.rerun()

    if "swing_scan" not in st.session_state:
        st.info("Run a scan to score every stock in the universe.")
        return
    report, data = st.session_state["swing_scan"]
    if report.source == "sample":
        st.warning("Synthetic sample data: not real stocks or prices.")
    render_scan(report, data, book, paper)


def render_scan(report: ScanReport, data, book: SwingBook, paper: PaperBook) -> None:
    longs = [s for s in report.scored if s.side == LONG]
    shorts = [s for s in report.scored if s.side == SHORT]
    c = st.columns(5)
    c[0].metric("Scanned", len(report.scored))
    c[1].metric("Long setups", len(longs))
    c[2].metric("Short setups", len(shorts))
    c[3].metric("Failed to load", len(report.failed))
    c[4].metric("NIFTY 60-day", f"{report.benchmark_ret_60:+.1%}")

    for alert in report.exit_alerts:
        cols = st.columns([6, 1])
        cols[0].warning(f"Exit alert · {alert.symbol} ({alert.side}): {alert.reason}, score "
                        f"{alert.score:+.2f}")
        if cols[1].button("Close", key=f"exit-{alert.side}-{alert.position_id}"):
            if alert.side == LONG:
                book.close(alert.position_id)
            else:
                paper.close(alert.position_id)
            st.rerun()

    left, right = st.columns(2)
    with left:
        st.subheader("Long ideas · delivery buys")
        for idea in [i for i in report.ideas if i.side == LONG]:
            idea_card(idea)
    with right:
        st.subheader("Short ideas · bear put spreads")
        shorts_ideas = [i for i in report.ideas if i.side == SHORT]
        if not shorts_ideas:
            st.caption("None this scan.")
        for idea in shorts_ideas:
            idea_card(idea)

    st.subheader("All stocks")
    table = pd.DataFrame([{
        "symbol": s.symbol, "name": s.name, "industry": s.industry, "close": s.close,
        "score": s.score, "side": s.side or "", "trend": s.agents["trend"]["score"],
        "momentum": s.agents["momentum"]["score"], "volume": s.agents["volume"]["score"],
        "RS 60d": s.rs_60, "above 200 EMA": s.above_200, "F&O": s.fno,
        "news": s.agents.get("news", {}).get("score"), "avg value ₹cr": s.avg_value_cr,
        "why": s.reason,
    } for s in report.scored])
    f1, f2 = st.columns(2)
    side = f1.selectbox("Show", ["All", "Long setups", "Short setups", "No setup"])
    industries = sorted(table.industry.unique())
    pick = f2.multiselect("Industry", industries)
    if side != "All":
        table = table[table.side == {"Long setups": LONG, "Short setups": SHORT,
                                     "No setup": ""}[side]]
    if pick:
        table = table[table.industry.isin(pick)]
    st.dataframe(table, hide_index=True, use_container_width=True, height=380)

    st.subheader("Chart")
    symbols = [i.symbol for i in report.ideas] + [s.symbol for s in report.scored
                                                  if s.symbol not in {i.symbol for i in
                                                                      report.ideas}]
    sym = st.selectbox("Stock", symbols, key="swing_chart_symbol")
    if sym:
        idea = next((i for i in report.ideas if i.symbol == sym), None)
        try:
            st.plotly_chart(price_chart(data.daily(sym), idea), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load {sym}: {exc}")
        score = next((s for s in report.scored if s.symbol == sym), None)
        if score:
            st.caption(score.rationale)


def review_tab(book: SwingBook, paper: PaperBook) -> None:
    st.caption("Approving opens a paper position: longs in the swing book, bearish spreads "
               "in the options paper book. No real orders are placed.")
    pending = book.ideas(PENDING)
    if not pending:
        st.info("Nothing waiting for review.")
    for row in pending:
        idea: SwingIdea = row["idea"]
        idea_card(idea)
        note = st.text_input("Note", key=f"swnote-{idea.id}")
        a, r, _ = st.columns([1, 1, 6])
        if a.button("Approve", key=f"swok-{idea.id}"):
            book.decide(idea.id, True, note, paper=paper)
            st.rerun()
        if r.button("Reject", key=f"swno-{idea.id}"):
            book.decide(idea.id, False, note)
            st.rerun()

    st.subheader("Idea log")
    rows = book.ideas(limit=300)
    if rows:
        st.dataframe(pd.DataFrame([{
            "time": r["created_at"][:16], "symbol": r["symbol"], "side": r["idea"].side,
            "status": r["status"], "score": r["idea"].stock.score,
            "max loss ₹": round(r["idea"].max_loss), "note": r["note"] or "",
        } for r in rows]), hide_index=True, use_container_width=True)


def positions_tab(settings: Settings, book: SwingBook, paper: PaperBook) -> None:
    today = date.today()
    state = paper.kill_switch()
    if state.get("active"):
        st.error(f"Kill switch tripped: {state.get('reason')}")
        if st.button("Reset kill switch", key="swing_reset"):
            paper.reset_kill_switch()
            st.rerun()
    longs = book.positions()
    spreads = [p for p in paper.positions() if p["underlying"] not in NSE_INDEX_NAMES]
    open_longs = [p for p in longs if p["status"] == "open"]
    c = st.columns(4)
    c[0].metric("Open longs", len(open_longs))
    c[1].metric("Open bearish spreads", sum(1 for p in spreads if p["status"] == "open"))
    c[2].metric("Swing P&L today", f"₹{book.daily_pnl(today):+,.0f}")
    realized = sum(p["pnl"] or 0 for p in longs) + sum(p["realized_pnl"] or 0 for p in spreads)
    c[3].metric("Realized (all time)", f"₹{realized:+,.0f}")
    st.caption("Positions update on every scan: longs walk through each new daily bar "
               "(stop, then target, then time stop); spreads are re-priced from the chain.")

    st.subheader("Long positions")
    if longs:
        st.dataframe(pd.DataFrame([{
            "#": p["id"], "symbol": p["symbol"], "status": p["status"], "qty": p["qty"],
            "entry": p["entry"], "stop": p["stop"], "target": p["target"],
            "last": p["last_close"], "sessions": p["sessions_held"],
            "P&L ₹": round(p["pnl"] if p["status"] == "closed" else p["unrealized_pnl"]),
            "exit": p["exit_reason"] or "",
        } for p in longs]), hide_index=True, use_container_width=True)
    else:
        st.caption("None yet.")
    st.subheader("Bearish spreads")
    if spreads:
        st.dataframe(pd.DataFrame([{
            "#": p["id"], "stock": p["underlying"], "status": p["status"],
            "legs": " / ".join(f"{leg.action} {leg.strike:g}{leg.option_type}"
                               for leg in p["legs"]),
            "expiry": p["expiry"], "entry ₹": round(p["entry_cost"]),
            "P&L ₹": round(p["realized_pnl"] if p["status"] == "closed"
                           else p["unrealized_pnl"]),
        } for p in spreads]), hide_index=True, use_container_width=True)
    else:
        st.caption("None yet.")


def backtest_tab() -> None:
    from fno_research.backtest import backtest_technical

    if "swing_scan" not in st.session_state:
        st.info("Run a scan first, then pick a stock to backtest.")
        return
    report, data = st.session_state["swing_scan"]
    sym = st.selectbox("Stock", [s.symbol for s in report.scored], key="swing_bt_symbol")
    st.caption("Walk-forward test of the technical agents on this stock's daily history "
               "(same method as the F&O backtest).")
    if st.button("Backtest", key="swing_bt"):
        with st.spinner(f"Walking forward through {sym}…"):
            result = backtest_technical(data.daily(sym))
        st.dataframe(result.summary, hide_index=True, use_container_width=True)


def swing_desk(settings: Settings) -> None:
    source = swing_sidebar(settings)
    book, paper = _books(str(settings.db_path))
    scan, review, positions, backtest = st.tabs([
        "Scanner", f"Review queue ({len(book.ideas(PENDING))})", "Positions", "Backtest",
    ])
    with scan:
        scanner_tab(settings, source, book, paper)
    with review:
        review_tab(book, paper)
    with positions:
        positions_tab(settings, book, paper)
    with backtest:
        backtest_tab()
