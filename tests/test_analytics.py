import math

import pytest

from fno_research.analytics.indicators import candles_to_frame, rsi, supertrend
from fno_research.analytics.options import (
    bs_price,
    fill_implied_vols,
    implied_vol,
    max_pain,
    oi_walls,
    pcr,
)
from fno_research.data.sample import SampleDataProvider


def test_implied_vol_round_trips_black_scholes():
    for opt in ("CE", "PE"):
        price = bs_price(25_000, 25_200, 7 / 365, 0.14, opt)
        assert implied_vol(price, 25_000, 25_200, 7 / 365, opt) == pytest.approx(0.14, abs=1e-4)


def test_implied_vol_rejects_price_below_intrinsic():
    assert implied_vol(10, 25_000, 24_000, 7 / 365, "CE") is None


def test_put_call_parity():
    t, r = 30 / 365, 0.065
    call = bs_price(25_000, 25_000, t, 0.15, "CE", r)
    put = bs_price(25_000, 25_000, t, 0.15, "PE", r)
    assert call - put == pytest.approx(25_000 - 25_000 * math.exp(-r * t), rel=1e-6)


def test_chain_metrics(chain):
    assert pcr(chain) > 1  # sample chain has heavier put OI
    walls = oi_walls(chain)
    assert walls.put_wall < chain.spot < walls.call_wall
    assert walls.put_wall <= max_pain(chain) <= walls.call_wall


def test_fill_implied_vols_recovers_sample_vols(chain):
    fill_implied_vols(chain)
    atm = chain.get(chain.atm_strike(), "CE")
    assert atm.iv == pytest.approx(0.13, abs=0.01)


def test_rsi_extremes():
    up = SampleDataProvider(drift=0.01).candles("NIFTY", "day", 60)
    down = SampleDataProvider(drift=-0.01).candles("NIFTY", "day", 60)
    assert rsi(candles_to_frame(up)["close"]).iloc[-1] > 70
    assert rsi(candles_to_frame(down)["close"]).iloc[-1] < 30


def test_supertrend_follows_trend():
    up = candles_to_frame(SampleDataProvider(drift=0.01).candles("NIFTY", "day", 120))
    down = candles_to_frame(SampleDataProvider(drift=-0.01).candles("NIFTY", "day", 120))
    assert supertrend(up).iloc[-1] == 1
    assert supertrend(down).iloc[-1] == -1
