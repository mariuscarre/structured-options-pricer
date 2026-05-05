"""Simple volatility strategies built from vanilla option prices."""

from __future__ import annotations

from core.black_scholes import call_price, put_price


def straddle_price(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Long straddle: long call + long put at the same strike."""
    return call_price(spot, strike, rate, volatility, time_to_expiry) + put_price(
        spot, strike, rate, volatility, time_to_expiry
    )


def strangle_price(
    spot: float,
    put_strike: float,
    call_strike: float,
    rate: float,
    volatility: float,
    time_to_expiry: float,
) -> float:
    """Long strangle: long OTM put + long OTM call."""
    if put_strike >= call_strike:
        raise ValueError("put_strike should be less than call_strike for a standard strangle")

    return put_price(spot, put_strike, rate, volatility, time_to_expiry) + call_price(
        spot, call_strike, rate, volatility, time_to_expiry
    )
