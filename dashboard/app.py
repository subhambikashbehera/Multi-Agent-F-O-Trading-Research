"""Streamlit dashboard: research runs, the human review queue, paper trading and backtests.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fno_research.analytics.buildup import oi_buildup
from fno_research.config import SUPPORTED_UNDERLYINGS, Settings
from fno_research.factory import build_market, build_pipeline
from fno_research.models import Direction, OptionChain, ResearchReport, TradeIdea
from fno_research.paper import PaperBook
from fno_research.review import BLOCKED, NO_TRADE, PENDING, ReviewQueue, decision_log

st.set_page_config(page_title="Research Desk", layout="wide")

DIRECTION_COLOR = {Direction.BULLISH: "green", Direction.BEARISH: "red", Direction.NEUTRAL: "gray"}
AGENT_LABEL = {
    "trend": "Trend (MA cross, ADX)",
    "momentum": "Momentum (RSI, MACD)",
    "volatility": "Volatility (ATR, Bollinger)",
    "volume": "Volume (OBV, VWAP)",
    "options_positioning": "Options positioning",
    "flows": "FII/DII flows",
    "news": "News (Claude)",
}
SOURCES = {
    "NSE website + Yahoo (free)": "nse",
    "Kite Connect": "kite",
    "Sample (synthetic)": "sample",
}


@st.cache_resource
def get_queue(path: str) -> ReviewQueue:
    return ReviewQueue(Path(path))


@st.cache_resource
def get_book(path: str) -> PaperBook:
    return PaperBook(Path(path))


# -- sidebar ------------------------------------------------------------------------------


def kite_login_panel(settings: Settings) -> None:
    from fno_research.data.kite import exchange_request_token, login_url

    st.sidebar.warning("No Kite access token for today.")
    if not settings.kite_api_key:
        st.sidebar.caption("Set KITE_API_KEY and KITE_API_SECRET in .env first.")
        return
    st.sidebar.markdown(f"1. [Log in to Kite]({login_url(settings)})")
    request_token = st.sidebar.text_input("2. Paste request_token from the redirect URL")
    if request_token and st.sidebar.button("Get access token"):
        try:
            token = exchange_request_token(settings, request_token.strip())
        except Exception as exc:
            st.sidebar.error(f"Login failed: {exc}")
            return
        st.session_state["kite_access_token"] = token
        st.sidebar.success("Logged in for today. Add KITE_ACCESS_TOKEN to .env to skip this.")
        st.rerun()


def sidebar() -> tuple[Settings, str, str]:
    settings = Settings()
    if st.session_state.get("kite_access_token"):
        settings.kite_access_token = st.session_state["kite_access_token"]

    st.sidebar.title("F&O Research Desk")
    labels = list(SOURCES)
    default = next((i for i, k in enumerate(labels) if SOURCES[k] == settings.data_source), 0)
    source = SOURCES[st.sidebar.radio("Market data", labels, index=default)]
    if source == "kite" and not settings.has_kite:
        kite_login_panel(settings)
    underlying = st.sidebar.selectbox("Underlying", SUPPORTED_UNDERLYINGS)

    with st.sidebar.expander("Weights (placeholders until backtested)"):
        g = settings.weights.group
        g["technical"] = st.slider("Technical group", 0.0, 1.0, g["technical"], 0.05)
        g["context"] = st.slider("Context group", 0.0, 1.0, g["context"], 0.05)
        st.caption("Agents within each group")
        for name, value in settings.weights.agent.items():
            settings.weights.agent[name] = st.slider(AGENT_LABEL[name], 0.0, 1.0, value, 0.05)
        a = settings.aggregator
        a.conflict_threshold = st.slider("Veto: conflict threshold", 0.0, 1.0,
                                         a.conflict_threshold, 0.05)

    with st.sidebar.expander("Risk limits"):
        r = settings.risk
        r.capital = st.number_input("Capital (₹)", 10_000.0, 1e9, float(r.capital), 50_000.0)
        r.max_risk_pct = st.number_input("Max risk per trade (%)", 0.1, 10.0,
                                         float(r.max_risk_pct), 0.1)
        r.daily_loss_pct = st.number_input("Daily loss limit (%)", 0.1, 20.0,
                                           float(r.daily_loss_pct), 0.1)
        r.max_lots = int(st.number_input("Max lots (keep 1 while testing)", 1, 100,
                                         int(r.max_lots)))
        r.min_confidence = st.slider("Min confidence", 0.0, 1.0, r.min_confidence, 0.05)
        r.min_agreement = st.slider("Min agreement", 0.0, 1.0, r.min_agreement, 0.05)

    if not settings.has_anthropic:
        st.sidebar.info("ANTHROPIC_API_KEY not set: the news agent is skipped.")
    return settings, underlying, source


# -- charts -------------------------------------------------------------------------------


def oi_chart(chain: OptionChain) -> go.Figure:
    frame = pd.DataFrame([q.model_dump() for q in chain.quotes])
    calls = frame[frame.option_type == "CE"]
    puts = frame[frame.option_type == "PE"]
    fig = go.Figure()
    fig.add_bar(x=calls.strike, y=calls.oi, name="Call OI", marker_color="#d62728")
    fig.add_bar(x=puts.strike, y=puts.oi, name="Put OI", marker_color="#2ca02c")
    fig.add_vline(x=chain.spot, line_dash="dash", annotation_text=f"Spot {chain.spot:,.0f}")
    fig.update_layout(barmode="group", height=360, margin=dict(t=30, b=10),
                      xaxis_title="Strike", yaxis_title="Open interest")
    return fig


def payoff_chart(idea: TradeIdea, spot: float) -> go.Figure:
    strikes = [leg.strike for leg in idea.legs]
    lo, hi = min(strikes + [spot]) * 0.97, max(strikes + [spot]) * 1.03
    settle = np.linspace(lo, hi, 200)
    pnl = np.zeros_like(settle)
    for leg in idea.legs:
        intrinsic = (np.maximum(settle - leg.strike, 0) if leg.option_type == "CE"
                     else np.maximum(leg.strike - settle, 0))
        sign = 1 if leg.action == "BUY" else -1
        pnl += sign * (intrinsic - leg.price) * leg.lots * idea.lot_size
    fig = go.Figure(go.Scatter(x=settle, y=pnl, mode="lines", name="P&L at expiry"))
    fig.add_hline(y=0, line_color="gray")
    fig.add_vline(x=spot, line_dash="dash", annotation_text="Spot")
    fig.update_layout(height=300, margin=dict(t=30, b=10), xaxis_title="Index at expiry",
                      yaxis_title="P&L (₹)")
    return fig


# -- research tab --------------------------------------------------------------------------


def agent_card(sig, contribution: float) -> None:
    with st.container(border=True):
        st.markdown(f"**{AGENT_LABEL.get(sig.agent, sig.agent)}**")
        st.markdown(f":{DIRECTION_COLOR[sig.direction]}[{sig.direction.value}] · "
                    f"score {sig.score:+.2f} · conf {sig.confidence:.2f} · "
                    f"contrib {contribution:+.3f}")
        st.caption(sig.rationale)
        if sig.features:
            with st.expander("Features"):
                st.json(sig.features)


def show_report(report: ResearchReport, chain: OptionChain | None, source: str) -> None:
    if source == "sample":
        st.warning("Synthetic sample data: not real market prices.")
    agg, ctx = report.aggregate, report.context
    color = DIRECTION_COLOR[agg.direction]
    st.markdown(f"### {report.underlying} · :{color}[{agg.direction.value.upper()}]")
    st.write(report.summary)
    if agg.veto:
        st.error(f"Conflict veto: {agg.veto_reason}")

    failed = [s for s in report.signals if s.rationale.startswith("Agent failed")]
    if chain is None and len(failed) >= len(report.signals) - 2:
        st.error("Market data unreachable, so this run has no signal. First error: "
                 + (failed[0].rationale if failed else "unknown"))
    cols = st.columns(5)
    cols[0].metric("Spot", f"{report.spot:,.2f}" if report.spot else "—")
    cols[1].metric("Score", f"{agg.score:+.2f}")
    cols[2].metric("Confidence", f"{agg.confidence:.2f}")
    cols[3].metric("Agreement", f"{agg.agreement:.2f}")
    cols[4].metric("Allocation multiplier", f"{agg.allocation_multiplier:.2f}")

    if ctx:
        cols = st.columns(4)
        vix = f"{ctx.vix:.2f}" if ctx.vix else "—"
        pct = f" · {ctx.vix_percentile:.0f}th pct" if ctx.vix_percentile is not None else ""
        cols[0].metric("India VIX", vix, help="Percentile versus the past year")
        cols[0].caption(f"Regime: **{ctx.vol_regime}**{pct}")
        cols[1].metric("Days to expiry", ctx.days_to_expiry if ctx.days_to_expiry is not None
                       else "—")
        cols[1].caption(ctx.expiry_tag.replace("_", " "))
        if ctx.flows:
            last = ctx.flows[-1]
            cols[2].metric(f"FII net {last.date:%d %b}", f"₹{last.fii_net:+,.0f} cr")
            cols[3].metric(f"DII net {last.date:%d %b}", f"₹{last.dii_net:+,.0f} cr")

    for group in ("technical", "context"):
        members = [s for s in report.signals if s.group == group]
        gs = agg.groups.get(group)
        head = f"{group.capitalize()} agents"
        if gs:
            head += (f" · :{DIRECTION_COLOR[gs.direction]}[{gs.direction.value}] "
                     f"score {gs.score:+.2f}, conf {gs.confidence:.2f}")
        st.subheader(head)
        for col, sig in zip(st.columns(len(members)), members, strict=True):
            with col:
                agent_card(sig, agg.contributions.get(sig.agent, 0))

    if chain is not None:
        st.subheader(f"Option chain OI · expiry {chain.expiry:%d %b %Y} · lot {chain.lot_size}")
        st.plotly_chart(oi_chart(chain), use_container_width=True)
        buildup = oi_buildup(chain)
        st.markdown(f"**OI build-up near the money** · calls: {buildup.call_state} · "
                    f"puts: {buildup.put_state} · read {buildup.score:+.2f}")
        with st.expander("Per-strike build-up"):
            table = pd.DataFrame(buildup.rows).pivot(
                index="strike", columns="side", values="state"
            ).rename(columns={"CE": "Calls", "PE": "Puts"})
            st.dataframe(table, use_container_width=True)
            st.caption("Price up + OI up = long build-up · price down + OI up = short "
                       "build-up · price up + OI down = short covering · price down + OI "
                       "down = long unwinding. Put writing is bullish, call writing bearish.")

    if report.idea:
        idea = report.idea
        st.subheader(f"Trade idea: {idea.strategy}")
        st.write(idea.rationale)
        st.dataframe(pd.DataFrame([leg.model_dump() for leg in idea.legs]),
                     hide_index=True, use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Max loss", f"₹{idea.max_loss:,.0f}")
        m2.metric("Max profit", f"₹{idea.max_profit:,.0f}" if idea.max_profit else "Unbounded")
        m3.metric("Breakeven", f"{idea.breakeven:,.2f}" if idea.breakeven else "—")
        st.plotly_chart(payoff_chart(idea, report.spot), use_container_width=True)

    if report.risk:
        st.subheader("Risk checks: " + ("passed" if report.risk.approved else "blocked"))
        for check in report.risk.checks:
            st.markdown(f"{'✅' if check.passed else '❌'} **{check.name}**: {check.detail}")


def research_tab(settings: Settings, underlying: str, source: str, queue: ReviewQueue) -> None:
    ready = source != "kite" or settings.has_kite
    if st.button(f"Run research on {underlying}", type="primary", disabled=not ready):
        with st.spinner("Running agents…"):
            try:
                pipeline = build_pipeline(settings, source)
                report = pipeline.run(underlying)
            except Exception as exc:
                st.error(f"Run failed: {exc}")
                return
        status = queue.add(report)
        st.session_state["last"] = (report, pipeline.last_chain, source)
        st.session_state["flash"] = {
            PENDING: ("success", "Idea passed risk and is waiting in the review queue."),
            BLOCKED: ("info", "Idea was blocked by the risk engine; logged for reference."),
            NO_TRADE: ("info", "No trade this run; logged for reference."),
        }[status]
        st.rerun()  # refresh the review-queue count in the tab label
    if "flash" in st.session_state:
        kind, message = st.session_state.pop("flash")
        getattr(st, kind)(message)
    if not ready:
        st.caption("Log in to Kite in the sidebar or pick another data source.")
    if "last" in st.session_state:
        show_report(*st.session_state["last"])


# -- review tab ----------------------------------------------------------------------------


def review_tab(queue: ReviewQueue, book: PaperBook) -> None:
    st.caption("Approving opens a paper position. This app never places real orders.")
    pending = queue.list(PENDING)
    if not pending:
        st.info("Nothing waiting for review.")
    for row in pending:
        report: ResearchReport = row["report"]
        idea = report.idea
        with st.container(border=True):
            st.markdown(f"**{report.underlying} · {idea.strategy}** · {row['created_at'][:16]}")
            st.write(report.summary)
            legs = ", ".join(f"{leg.action} {leg.lots}× {leg.tradingsymbol} @ {leg.price}"
                             for leg in idea.legs)
            st.caption(f"{legs} · max loss ₹{idea.max_loss:,.0f}")
            note = st.text_input("Note", key=f"note-{row['id']}")
            a, r, _ = st.columns([1, 1, 6])
            if a.button("Approve", key=f"ok-{row['id']}"):
                book.open_from_report(queue.decide(row["id"], True, note))
                st.rerun()
            if r.button("Reject", key=f"no-{row['id']}"):
                queue.decide(row["id"], False, note)
                st.rerun()

    st.subheader("Signal and decision log")
    log = decision_log(queue, book)
    if log:
        traded = [r for r in log if r["paper"] == "closed"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Runs logged", len(log))
        c2.metric("Approved → paper", sum(1 for r in log if r["paper"]))
        c3.metric("Closed trades won",
                  f"{sum(1 for r in traded if r['pnl'] > 0)}/{len(traded)}" if traded else "—")
        st.dataframe(pd.DataFrame(log), hide_index=True, use_container_width=True)


# -- paper tab -----------------------------------------------------------------------------


def paper_tab(settings: Settings, book: PaperBook) -> None:
    state = book.kill_switch()
    if state.get("active"):
        st.error(f"Kill switch tripped at {state['tripped_at'][:16]}: {state['reason']} "
                 "Every new idea is blocked until you reset it.")
        if st.button("Reset kill switch"):
            book.reset_kill_switch()
            st.rerun()
    else:
        st.success(f"Kill switch off · daily loss limit ₹{settings.risk.daily_loss_limit:,.0f}")

    today = pd.Timestamp.now().date()
    positions = book.positions()
    open_pos = [p for p in positions if p["status"] == "open"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Open positions", len(open_pos))
    c2.metric("Today's P&L", f"₹{book.daily_pnl(today):+,.0f}")
    c3.metric("Realized P&L (all time)",
              f"₹{sum(p['realized_pnl'] or 0 for p in positions):+,.0f}")
    st.caption("Positions are marked to market on every research run for their underlying.")

    for p in open_pos:
        with st.container(border=True):
            legs = ", ".join(f"{leg.action} {leg.lots}× {leg.tradingsymbol} @ {leg.price}"
                             for leg in p["legs"])
            st.markdown(f"**#{p['id']} {p['underlying']} {p['strategy']}** · expiry "
                        f"{p['expiry']:%d %b} · unrealized ₹{p['unrealized_pnl']:+,.0f}")
            st.caption(f"{legs} · last marked {p['last_marked_at'][:16]}")
            if st.button("Close at last mark", key=f"close-{p['id']}"):
                book.close(p["id"])
                st.rerun()

    closed = [p for p in positions if p["status"] == "closed"]
    if closed:
        st.subheader("Closed")
        st.dataframe(pd.DataFrame([
            {"id": p["id"], "underlying": p["underlying"], "strategy": p["strategy"],
             "opened": p["opened_at"][:16], "closed": p["closed_at"][:16],
             "entry ₹": round(p["entry_cost"]), "exit ₹": round(p["exit_value"]),
             "P&L ₹": round(p["realized_pnl"])}
            for p in closed
        ]), hide_index=True, use_container_width=True)


# -- backtest tab --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def run_backtest(source: str, underlying: str, years: float, weights_key: str):
    from fno_research.analytics.indicators import candles_to_frame
    from fno_research.backtest import backtest_technical

    settings = Settings()
    market = build_market(settings, source)
    daily = candles_to_frame(market.candles(underlying, "day", int(365 * years) + 200))
    return backtest_technical(daily, weights=settings.weights)


def backtest_tab(underlying: str, source: str) -> None:
    st.caption(
        "Walk-forward test of each technical agent on daily history: on each day the agent "
        "sees only data up to that close, and its score is compared with the index return "
        "over the next 1/3/5 sessions. Compare hit rate with base_up_rate and edge_bps with "
        "market_bps (just holding): an agent earns trust with edge and IC that beat those "
        "consistently, not a lucky month."
    )
    years = st.slider("Years of history", 1.0, 5.0, 2.0, 0.5)
    if st.button(f"Backtest technical agents on {underlying}"):
        with st.spinner("Walking forward day by day…"):
            try:
                result = run_backtest(source, underlying, years, "default")
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                return
        st.session_state["backtest"] = (underlying, result)
    if "backtest" in st.session_state:
        name, result = st.session_state["backtest"]
        sig = result.signals
        st.markdown(f"**{name}** · {len(sig)} test days, "
                    f"{sig.index[0]:%d %b %Y} to {sig.index[-1]:%d %b %Y}")
        st.dataframe(result.summary, hide_index=True, use_container_width=True)
        five = result.summary[result.summary.horizon == 5]
        fig = go.Figure(go.Bar(x=five.agent, y=five.edge_bps))
        fig.update_layout(height=300, margin=dict(t=30, b=10),
                          yaxis_title="Avg 5-day edge (bps)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Context agents (positioning, flows, news) are scored from the feature "
                   "store once live runs have built up history.")


def fno_desk() -> None:
    settings, underlying, source = sidebar()
    queue = get_queue(str(settings.db_path))
    book = get_book(str(settings.db_path))
    research, review, paper, backtest = st.tabs([
        "Research", f"Review queue ({len(queue.list(PENDING))})", "Paper trading", "Backtest",
    ])
    with research:
        research_tab(settings, underlying, source, queue)
    with review:
        review_tab(queue, book)
    with paper:
        paper_tab(settings, book)
    with backtest:
        backtest_tab(underlying, source)


def main() -> None:
    # Desk switch across the top; each desk brings its own sidebar and tabs.
    desk = st.segmented_control("Desk", ["F&O", "Swing trading"], default="F&O",
                                key="desk", label_visibility="collapsed")
    if desk == "Swing trading":
        from swing_ui import swing_desk

        swing_desk(Settings())
    else:
        fno_desk()


main()
