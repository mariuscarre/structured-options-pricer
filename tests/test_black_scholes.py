import math

from core.black_scholes import call_price, put_price
from risk.greeks import call_delta, gamma, put_delta, vega


def test_black_scholes_prices_are_positive():
    call = call_price(100, 100, 0.05, 0.2, 1.0)
    put = put_price(100, 100, 0.05, 0.2, 1.0)
    assert call > 0
    assert put > 0


def test_put_call_parity():
    spot = 100
    strike = 100
    rate = 0.05
    vol = 0.2
    t = 1.0

    call = call_price(spot, strike, rate, vol, t)
    put = put_price(spot, strike, rate, vol, t)
    lhs = call - put
    rhs = spot - strike * math.exp(-rate * t)
    assert abs(lhs - rhs) < 1e-6


def test_basic_greeks_ranges():
    spot = 100
    strike = 100
    rate = 0.05
    vol = 0.2
    t = 1.0

    c_delta = call_delta(spot, strike, rate, vol, t)
    p_delta = put_delta(spot, strike, rate, vol, t)
    g = gamma(spot, strike, rate, vol, t)
    v = vega(spot, strike, rate, vol, t)

    assert 0.0 < c_delta < 1.0
    assert -1.0 < p_delta < 0.0
    assert g > 0.0
    assert v > 0.0
