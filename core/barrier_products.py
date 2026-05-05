"""Monte Carlo barrier products pricing with optional Brownian-bridge correction."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]
BarrierStyle = Literal["down-and-out", "up-and-out", "down-and-in", "up-and-in"]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def vanilla_black_scholes(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Vanilla Black-Scholes with continuous dividend yield."""
    if maturity <= 0.0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if volatility <= 0.0:
        fwd = spot * math.exp((rate - dividend_yield) * maturity)
        disc = math.exp(-rate * maturity)
        intrinsic = max(fwd - strike, 0.0) if option_type == "call" else max(strike - fwd, 0.0)
        return disc * intrinsic

    sqrt_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    disc_r = math.exp(-rate * maturity)
    disc_q = math.exp(-dividend_yield * maturity)

    if option_type == "call":
        return spot * disc_q * _normal_cdf(d1) - strike * disc_r * _normal_cdf(d2)
    return strike * disc_r * _normal_cdf(-d2) - spot * disc_q * _normal_cdf(-d1)


def _bridge_hit_probability(
    s0: np.ndarray,
    s1: np.ndarray,
    barrier: float,
    sigma: float,
    dt: float,
    is_down: bool,
) -> np.ndarray:
    """
    Crossing probability for a GBM step via log-space Brownian bridge approximation.
    Valid when both endpoints remain on non-hit side of barrier.
    """
    eps = 1e-12
    var = max(sigma * sigma * dt, eps)
    log_h = math.log(max(barrier, eps))
    log_s0 = np.log(np.clip(s0, eps, None))
    log_s1 = np.log(np.clip(s1, eps, None))

    if is_down:
        a = np.maximum(log_s0 - log_h, 0.0)
        b = np.maximum(log_s1 - log_h, 0.0)
    else:
        a = np.maximum(log_h - log_s0, 0.0)
        b = np.maximum(log_h - log_s1, 0.0)

    p = np.exp(-2.0 * a * b / var)
    return np.clip(p, 0.0, 1.0)


def price_barrier_option_mc(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    barrier: float,
    option_type: OptionType,
    barrier_type: BarrierStyle,
    n_simulations: int = 50_000,
    n_steps: int = 252,
    rebate: float = 0.0,
    dividend_yield: float = 0.0,
    use_brownian_bridge: bool = True,
    seed: int | None = 42,
    return_payoffs: bool = False,
) -> dict[str, float]:
    """Price a barrier option by path simulation under risk-neutral GBM."""
    if spot <= 0.0 or strike <= 0.0 or barrier <= 0.0:
        raise ValueError("Spot, strike, and barrier must be > 0.")
    if maturity <= 0.0:
        raise ValueError("Maturity must be > 0.")
    if volatility <= 0.0:
        raise ValueError("Volatility must be > 0.")
    if n_simulations < 1000:
        raise ValueError("n_simulations must be >= 1000.")
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2.")

    is_down = "down" in barrier_type
    is_out = "out" in barrier_type

    if is_down and barrier >= spot:
        raise ValueError("Down barriers must be below spot.")
    if (not is_down) and barrier <= spot:
        raise ValueError("Up barriers must be above spot.")

    rng = np.random.default_rng(seed)
    dt = maturity / float(n_steps)
    sqrt_dt = math.sqrt(dt)
    drift = (rate - dividend_yield - 0.5 * volatility * volatility) * dt
    vol_step = volatility * sqrt_dt

    s = np.full(n_simulations, spot, dtype=float)
    hit = np.zeros(n_simulations, dtype=bool)

    for _ in range(n_steps):
        z = rng.standard_normal(n_simulations)
        s_next = s * np.exp(drift + vol_step * z)

        direct_hit = (s_next <= barrier) if is_down else (s_next >= barrier)
        step_hit = direct_hit.copy()

        if use_brownian_bridge:
            # For unresolved paths where endpoint check missed crossing.
            unresolved = ~step_hit
            if np.any(unresolved):
                if is_down:
                    bridge_mask = unresolved & (s > barrier) & (s_next > barrier)
                else:
                    bridge_mask = unresolved & (s < barrier) & (s_next < barrier)

                if np.any(bridge_mask):
                    p_hit = _bridge_hit_probability(
                        s[bridge_mask], s_next[bridge_mask], barrier, volatility, dt, is_down=is_down
                    )
                    u = rng.random(np.count_nonzero(bridge_mask))
                    bridge_hit = u < p_hit
                    idx = np.where(bridge_mask)[0]
                    step_hit[idx] = bridge_hit

        hit |= step_hit
        s = s_next

    if option_type == "call":
        vanilla_payoff = np.maximum(s - strike, 0.0)
    else:
        vanilla_payoff = np.maximum(strike - s, 0.0)

    if is_out:
        payoff = np.where(hit, rebate, vanilla_payoff)
    else:
        payoff = np.where(hit, vanilla_payoff, rebate)

    disc = math.exp(-rate * maturity)
    discounted = payoff * disc
    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / math.sqrt(n_simulations))
    ci_95_low = price - 1.96 * std_error
    ci_95_high = price + 1.96 * std_error

    hit_prob = float(np.mean(hit))
    ki_prob = hit_prob
    ko_prob = 1.0 - hit_prob

    result = {
        "price": price,
        "std_error": std_error,
        "ci_95_low": float(ci_95_low),
        "ci_95_high": float(ci_95_high),
        "hit_probability": hit_prob,
        "knock_in_probability": ki_prob,
        "knock_out_probability": ko_prob,
        "discounted_payoff_mean": price,
    }
    if return_payoffs:
        result["discounted_payoffs"] = discounted
    return result


def product_payoff_explanation(option_type: str, barrier_type: str, rebate: float) -> str:
    """Short readable payoff explanation for selected product."""
    direction = "drops to or below" if "down" in barrier_type else "rises to or above"
    knockout_text = (
        f"dies if spot {direction} the barrier; rebate={rebate:.2f} paid when knocked out"
        if "out" in barrier_type
        else f"activates only if spot {direction} the barrier; rebate={rebate:.2f} paid if never activated"
    )
    intrinsic = "max(S_T-K,0)" if option_type == "call" else "max(K-S_T,0)"
    return f"Terminal payoff references {intrinsic}; barrier condition: option {knockout_text}."
