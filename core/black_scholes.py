"""Black-Scholes pricing for European call and put options."""

from __future__ import annotations

import math


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def d1(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Intermediate Black-Scholes term d1."""
    numerator = math.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_expiry
    denominator = volatility * math.sqrt(time_to_expiry)
    return numerator / denominator


def d2(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Intermediate Black-Scholes term d2."""
    return d1(spot, strike, rate, volatility, time_to_expiry) - volatility * math.sqrt(time_to_expiry)


def call_price(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Price a European call option with Black-Scholes."""
    _d1 = d1(spot, strike, rate, volatility, time_to_expiry)
    _d2 = d2(spot, strike, rate, volatility, time_to_expiry)
    return spot * normal_cdf(_d1) - strike * math.exp(-rate * time_to_expiry) * normal_cdf(_d2)


def put_price(spot: float, strike: float, rate: float, volatility: float, time_to_expiry: float) -> float:
    """Price a European put option with Black-Scholes."""
    _d1 = d1(spot, strike, rate, volatility, time_to_expiry)
    _d2 = d2(spot, strike, rate, volatility, time_to_expiry)
    return strike * math.exp(-rate * time_to_expiry) * normal_cdf(-_d2) - spot * normal_cdf(-_d1)
