"""Black-Scholes Greeks for European options."""

from __future__ import annotations

import math

from core.black_scholes import d1, d2, normal_cdf, normal_pdf


def call_delta(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Sensitivity of call price to spot."""
    return normal_cdf(d1(spot, strike, rate, volatility, time_to_expiry))


def put_delta(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Sensitivity of put price to spot."""
    return call_delta(spot, strike, rate, volatility, time_to_expiry) - 1.0


def gamma(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Sensitivity of delta to spot for both call and put."""
    _d1 = d1(spot, strike, rate, volatility, time_to_expiry)
    return normal_pdf(_d1) / (spot * volatility * math.sqrt(time_to_expiry))


def vega(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Sensitivity to volatility (per 1.00 volatility unit)."""
    _d1 = d1(spot, strike, rate, volatility, time_to_expiry)
    return spot * normal_pdf(_d1) * math.sqrt(time_to_expiry)


def call_theta(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Sensitivity to passage of time (per year)."""
    _d1 = d1(spot, strike, rate, volatility, time_to_expiry)
    _d2 = d2(spot, strike, rate, volatility, time_to_expiry)
    first_term = -(spot * normal_pdf(_d1) * volatility) / (2.0 * math.sqrt(time_to_expiry))
    second_term = -rate * strike * math.exp(-rate * time_to_expiry) * normal_cdf(_d2)
    return first_term + second_term


def put_theta(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Sensitivity to passage of time (per year)."""
    _d1 = d1(spot, strike, rate, volatility, time_to_expiry)
    _d2 = d2(spot, strike, rate, volatility, time_to_expiry)
    first_term = -(spot * normal_pdf(_d1) * volatility) / (2.0 * math.sqrt(time_to_expiry))
    second_term = rate * strike * math.exp(-rate * time_to_expiry) * normal_cdf(-_d2)
    return first_term + second_term


def call_rho(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Sensitivity of call price to interest rates."""
    _d2 = d2(spot, strike, rate, volatility, time_to_expiry)
    return strike * time_to_expiry * math.exp(-rate * time_to_expiry) * normal_cdf(_d2)


def put_rho(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Sensitivity of put price to interest rates."""
    _d2 = d2(spot, strike, rate, volatility, time_to_expiry)
    return -strike * time_to_expiry * math.exp(-rate * time_to_expiry) * normal_cdf(-_d2)
