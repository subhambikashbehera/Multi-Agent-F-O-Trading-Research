from datetime import date

import numpy as np
import pandas as pd

from fno_research.analytics.indicators import candles_to_frame
from fno_research.analytics.regime import expiry_tag, vol_regime
from fno_research.backtest import backtest_technical, evaluate_stored, score_series
from fno_research.data.sample import SampleDataProvider


def test_vol_regime_bands_and_percentile():
    assert vol_regime(None) == ("unknown", None)
    assert vol_regime(11)[0] == "low"
    assert vol_regime(15)[0] == "normal"
    assert vol_regime(22)[0] == "elevated"
    assert vol_regime(32)[0] == "extreme"
    history = list(np.linspace(10, 20, 100))
    regime, pct = vol_regime(19.9, history)
    assert regime == "extreme" and pct >= 90
    assert vol_regime(10.5, history)[0] == "low"


def test_expiry_tag():
    assert expiry_tag(date(2026, 9, 29), date(2026, 9, 29)) == (0, "expiry_day")
    assert expiry_tag(date(2026, 9, 29), date(2026, 9, 27)) == (2, "near_expiry")
    assert expiry_tag(date(2026, 9, 29), date(2026, 9, 22)) == (7, "mid_cycle")


def test_score_series_perfect_predictor():
    fwd = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    out = score_series((fwd * 30).clip(-1, 1), pd.Series([1.0] * 5), fwd, 0.15)
    assert out["hit_rate"] == 1.0 and out["ic"] == 1.0 and out["edge_bps"] > 0


def test_backtest_has_no_lookahead():
    frame = candles_to_frame(SampleDataProvider(drift=0.003).candles("NIFTY", "day", 200))
    result = backtest_technical(frame, warmup=120)
    # Changing the future must not change any past score.
    altered = frame.copy()
    altered.iloc[-10:, altered.columns.get_loc("close")] *= 1.2
    again = backtest_technical(altered, warmup=120)
    cols = [c for c in result.signals.columns if c.endswith("_score")]
    pd.testing.assert_frame_equal(result.signals[cols].iloc[:-10], again.signals[cols].iloc[:-10])
    assert set(result.summary.agent) == {"trend", "momentum", "volatility", "volume",
                                         "technical_group"}


def test_evaluate_stored_with_empty_store(settings):
    from fno_research.store import FeatureStore

    store = FeatureStore(settings.db_path)
    frame = candles_to_frame(SampleDataProvider().candles("NIFTY", "day", 100))
    assert evaluate_stored(store, "NIFTY", "flows", frame).empty
