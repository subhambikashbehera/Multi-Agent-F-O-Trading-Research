"""FII/DII cash-market flows. Slow-moving context, so confidence stays modest."""

from __future__ import annotations

from fno_research.agents.base import clamp, make_signal
from fno_research.models import AgentSignal, FlowSnapshot

# ₹ crore of net FII buying over the window that counts as a full-strength signal.
FULL_SCALE_CRORE = 10_000


class FlowsAgent:
    name = "flows"
    group = "context"

    def analyse_flows(self, flows: list[FlowSnapshot]) -> AgentSignal:
        if not flows:
            return make_signal(self.name, 0, 0, "No FII/DII data.", {}, self.group)
        recent = sorted(flows, key=lambda f: f.date)[-5:]
        fii = sum(f.fii_net for f in recent)
        dii = sum(f.dii_net for f in recent)
        score = clamp(fii / FULL_SCALE_CRORE)
        # One day of flows is noise; confidence builds as the window fills.
        confidence = 0.1 + 0.3 * len(recent) / 5
        if fii * dii < 0 and abs(dii) > abs(fii):
            confidence *= 0.7  # domestic buying is absorbing FII selling (or vice versa)
        rationale = (
            f"FII net ₹{fii:+,.0f} cr, DII net ₹{dii:+,.0f} cr over the last "
            f"{len(recent)} session(s) to {recent[-1].date:%d %b}."
        )
        return make_signal(self.name, score, confidence, rationale, {
            "fii_net_cr": round(fii, 1), "dii_net_cr": round(dii, 1), "days": len(recent),
        }, self.group)
