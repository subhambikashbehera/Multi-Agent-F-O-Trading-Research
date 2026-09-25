"""Market-context tags: volatility regime from India VIX and expiry proximity."""

from __future__ import annotations

from datetime import date

# Fixed VIX bands, used when there is not enough history for a percentile.
VIX_BANDS = [(12.0, "low"), (18.0, "normal"), (25.0, "elevated")]


def vol_regime(vix: float | None, history: list[float] | None = None) -> tuple[str, float | None]:
    """Returns (regime, percentile). Percentile is versus `history` when it has >= 60 points."""
    if vix is None:
        return "unknown", None
    if history and len(history) >= 60:
        pct = 100 * sum(1 for v in history if v <= vix) / len(history)
        if pct < 25:
            regime = "low"
        elif pct < 70:
            regime = "normal"
        elif pct < 90:
            regime = "elevated"
        else:
            regime = "extreme"
        # An absolute spike is extreme whatever the last year looked like.
        if vix >= 30:
            regime = "extreme"
        return regime, round(pct, 1)
    for bound, name in VIX_BANDS:
        if vix < bound:
            return name, None
    return ("extreme" if vix >= 30 else "elevated"), None


def expiry_tag(expiry: date, today: date) -> tuple[int, str]:
    days = (expiry - today).days
    if days <= 0:
        return days, "expiry_day"
    if days <= 2:
        return days, "near_expiry"
    return days, "mid_cycle"
