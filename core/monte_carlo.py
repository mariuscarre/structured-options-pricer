"""Monte Carlo pricing for European options under geometric Brownian motion."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]


def european_option_monte_carlo(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_expiry: float,
    n_simulations: int = 50_000,
    option_type: OptionType = "call",
    seed: int | None = None,
) -> float:
    """Price a European call or put via one-step Monte Carlo."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_simulations)

    drift = (rate - 0.5 * volatility**2) * time_to_expiry
    diffusion = volatility * math.sqrt(time_to_expiry) * z
    terminal_prices = spot * np.exp(drift + diffusion)

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - strike, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(strike - terminal_prices, 0.0)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    discount_factor = math.exp(-rate * time_to_expiry)
    return float(discount_factor * np.mean(payoffs))
