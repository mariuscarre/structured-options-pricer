import pytest

from core.black_scholes import call_price, put_price
from instruments.volatility_strategies import straddle_price, strangle_price


def test_straddle_matches_call_plus_put():
    spot = 100
    strike = 100
    rate = 0.05
    vol = 0.2
    t = 1.0

    strategy_price = straddle_price(spot, strike, rate, vol, t)
    manual_price = call_price(spot, strike, rate, vol, t) + put_price(spot, strike, rate, vol, t)
    assert abs(strategy_price - manual_price) < 1e-10


def test_strangle_matches_component_options():
    spot = 100
    put_strike = 95
    call_strike = 105
    rate = 0.05
    vol = 0.2
    t = 1.0

    strategy_price = strangle_price(spot, put_strike, call_strike, rate, vol, t)
    manual_price = put_price(spot, put_strike, rate, vol, t) + call_price(spot, call_strike, rate, vol, t)
    assert abs(strategy_price - manual_price) < 1e-10


def test_strangle_validates_strikes():
    with pytest.raises(ValueError):
        strangle_price(100, 105, 100, 0.05, 0.2, 1.0)
