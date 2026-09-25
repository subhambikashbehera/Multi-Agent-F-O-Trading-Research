"""Streamlit dashboard: run the research pipeline and work the human review queue.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fno_research.config import INDEX_SPOT_SYMBOLS, Settings
from fno_research.data.sample import SampleDataProvider, StaticNewsProvider
from fno_research.models import Direction, OptionChain, ResearchReport, TradeIdea
from fno_research.pipeline import ResearchPipeline
from fno_research.review import BLOCKED, NO_TRADE, PENDING, ReviewQueue

st.set_page_config(page_title="F&O Research Desk", layout="wide")

DIRECTION_COLOR = {Direction.BULLISH: "green", Direction.BEARISH: "red", Direction.NEUTRAL: "gray"}
AGENT_LABEL = {
    "price_action": "Price action",
    "options_positioning": "Options positioning",
    "news": "News (Claude)",
}


@st.cache_resource
def get_queue(path: str) -> ReviewQueue:
    from pathlib import Path

    return ReviewQueue(Path(path))


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


def sidebar() -> tuple[Settings, str, bool]:
    settings = Settings()
    if st.session_state.get("kite_access_token"):
        settings.kite_access_token = st.session_state["kite_access_token"]

    st.sidebar.title("F&O Research Desk")
    source = st.sidebar.radio("Market data", ["Kite Connect (live)", "Sample (synthetic)"])
    use_sample = source.startswith("Sample")
    if not use_sample and not settings.has_kite:
        kite_login_panel(settings)
    underlying = st.sidebar.selectbox("Underlying", list(INDEX_SPOT_SYMBOLS))

    with st.sidebar.expander("Agent weights"):
        w = settings.weights
        w.price_action = st.slider("Price action", 0.0, 1.0, w.price_action, 0.05)
        w.options_positioning = st.slider("Options positioning", 0.0, 1.0,
                                          w.options_positioning, 0.05)
        w.news = st.slider("News", 0.0, 1.0, w.news, 0.05)

    with st.sidebar.expander("Risk limits"):
        r = settings.risk
        r.capital = st.number_input("Capital (₹)", 10_000.0, 1e9, float(r.capital), 50_000.0)
        r.max_risk_pct = st.number_input("Max risk per trade (%)", 0.1, 10.0,
                                         float(r.max_risk_pct), 0.1)
        r.max_lots = int(st.number_input("Max lots", 1, 100, int(r.max_lots)))
        r.min_confidence = st.slider("Min confidence", 0.0, 1.0, r.min_confidence, 0.05)
        r.min_agreement = st.slider("Min agreement", 0.0, 1.0, r.min_agreement, 0.05)

    if not settings.has_anthropic:
        st.sidebar.info("ANTHROPIC_API_KEY not set: the news agent is skipped.")
    return settings, underlying, use_sample


def build(settings: Settings, use_sample: bool) -> ResearchPipeline:
    if use_sample:
        return ResearchPipeline(settings, SampleDataProvider(), StaticNewsProvider())
    from fno_research.data.kite import KiteDataProvider
    from fno_research.data.news import RSSNewsProvider

    return ResearchPipeline(settings, KiteDataProvider(settings), RSSNewsProvider())


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


def show_report(report: ResearchReport, chain: OptionChain | None, sample: bool) -> None:
    if sample:
        st.warning("Synthetic sample data: not real market prices.")
    agg = report.aggregate
    color = DIRECTION_COLOR[agg.direction]
    st.markdown(f"### {report.underlying} · :{color}[{agg.direction.value.upper()}]")
    st.write(report.summary)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"{report.spot:,.2f}")
    c2.metric("Score", f"{agg.score:+.2f}")
    c3.metric("Confidence", f"{agg.confidence:.2f}")
    c4.metric("Agreement", f"{agg.agreement:.2f}")

    st.subheader("Agents")
    for col, sig in zip(st.columns(len(report.signals)), report.signals, strict=True):
        with col.container(border=True):
            st.markdown(f"**{AGENT_LABEL.get(sig.agent, sig.agent)}**")
            st.markdown(f":{DIRECTION_COLOR[sig.direction]}[{sig.direction.value}] · "
                        f"score {sig.score:+.2f} · conf {sig.confidence:.2f} · "
                        f"contrib {agg.contributions.get(sig.agent, 0):+.3f}")
            st.caption(sig.rationale)
            if sig.features:
                with st.expander("Features"):
                    st.json(sig.features)

    if chain is not None:
        st.subheader(f"Option chain OI · expiry {chain.expiry:%d %b %Y} · lot {chain.lot_size}")
        st.plotly_chart(oi_chart(chain), use_container_width=True)

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


def research_tab(settings: Settings, underlying: str, use_sample: bool,
                 queue: ReviewQueue) -> None:
    ready = use_sample or settings.has_kite
    if st.button(f"Run research on {underlying}", type="primary", disabled=not ready):
        with st.spinner("Running agents…"):
            try:
                pipeline = build(settings, use_sample)
                report = pipeline.run(underlying)
            except Exception as exc:
                st.error(f"Run failed: {exc}")
                return
        status = queue.add(report)
        st.session_state["last"] = (report, pipeline.last_chain, use_sample)
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
        st.caption("Log in to Kite in the sidebar or switch to sample data.")
    if "last" in st.session_state:
        show_report(*st.session_state["last"])


def review_tab(queue: ReviewQueue) -> None:
    st.caption("Approving records your decision only. This app never places orders.")
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
                queue.decide(row["id"], True, note)
                st.rerun()
            if r.button("Reject", key=f"no-{row['id']}"):
                queue.decide(row["id"], False, note)
                st.rerun()

    st.subheader("History")
    history = [
        {
            "time": row["created_at"][:16],
            "underlying": row["underlying"],
            "status": row["status"],
            "direction": row["report"].aggregate.direction.value,
            "score": row["report"].aggregate.score,
            "strategy": row["report"].idea.strategy if row["report"].idea else "",
            "note": row["reviewer_note"] or "",
        }
        for row in queue.list(limit=100)
    ]
    if history:
        st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)


def main() -> None:
    settings, underlying, use_sample = sidebar()
    queue = get_queue(str(settings.db_path))
    research, review = st.tabs(["Research", f"Review queue ({len(queue.list(PENDING))})"])
    with research:
        research_tab(settings, underlying, use_sample, queue)
    with review:
        review_tab(queue)


main()
