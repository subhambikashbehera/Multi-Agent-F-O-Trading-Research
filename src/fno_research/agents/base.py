from __future__ import annotations

from fno_research.models import AgentSignal, Direction


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def make_signal(agent: str, score: float, confidence: float, rationale: str,
                features: dict, group: str = "", threshold: float = 0.15) -> AgentSignal:
    score = clamp(score)
    return AgentSignal(
        agent=agent,
        group=group,
        score=round(score, 4),
        confidence=round(clamp(confidence, 0.0, 1.0), 4),
        direction=Direction.from_score(score, threshold),
        rationale=rationale,
        features=features,
    )


def vote_confidence(votes: dict[str, tuple[float, float]], score: float,
                    floor: float = 0.25, full_scale: float = 0.5) -> float:
    """Confidence from how much of the vote weight agrees with the net score, and its size."""
    agree = sum(w for v, w in votes.values() if v * score > 0)
    return floor + (0.9 - floor) * agree * min(abs(score) / full_scale, 1.0)
