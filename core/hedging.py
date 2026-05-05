"""Delta hedging utilities for educational P&L simulations."""

from __future__ import annotations

from typing import Literal

import numpy as np

from core.black_scholes import call_price, put_price
from risk.greeks import call_delta, put_delta

OptionType = Literal["call", "put"]
PositionSide = Literal["long", "short"]


def signed_position(position: float, side: PositionSide) -> float:
    """Return positive quantity for long and negative for short."""
    if side not in {"long", "short"}:
        raise ValueError("side must be either 'long' or 'short'")
    return position if side == "long" else -position


def option_delta(
    option_type: OptionType,
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_expiry: float,
) -> float:
    """Return Black-Scholes delta for a call or put."""
    if option_type == "call":
        return call_delta(spot, strike, rate, volatility, time_to_expiry)
    if option_type == "put":
        return put_delta(spot, strike, rate, volatility, time_to_expiry)
    raise ValueError("option_type must be either 'call' or 'put'")


def hedge_quantity(delta: float, position: float, side: PositionSide) -> float:
    """Compute stock hedge quantity using: hedge = -delta * signed_position."""
    pos = signed_position(position, side)
    return -delta * pos


def option_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_expiry: float,
) -> float:
    """Return Black-Scholes price for a call or put."""
    if option_type == "call":
        return call_price(spot, strike, rate, volatility, time_to_expiry)
    if option_type == "put":
        return put_price(spot, strike, rate, volatility, time_to_expiry)
    raise ValueError("option_type must be either 'call' or 'put'")


def simulate_delta_hedge_pnl(
    option_type: OptionType,
    side: PositionSide,
    position: float,
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_expiry: float,
    min_move: float = -0.10,
    max_move: float = 0.10,
    n_points: int = 41,
) -> dict[str, np.ndarray | float]:
    """Simulate option, hedge, and total P&L across spot moves."""
    moves = np.linspace(min_move, max_move, n_points)
    spot_grid = spot * (1.0 + moves)

    base_delta = option_delta(option_type, spot, strike, rate, volatility, time_to_expiry)
    pos = signed_position(position, side)
    hedge_qty = -base_delta * pos

    base_price = option_price(option_type, spot, strike, rate, volatility, time_to_expiry)
    new_prices = np.array(
        [option_price(option_type, s, strike, rate, volatility, time_to_expiry) for s in spot_grid], dtype=float
    )

    option_pnl = (new_prices - base_price) * pos
    hedge_pnl = hedge_qty * (spot_grid - spot)
    total_pnl = option_pnl + hedge_pnl

    return {
        "moves_pct": moves * 100.0,
        "spot_grid": spot_grid,
        "option_pnl": option_pnl,
        "hedge_pnl": hedge_pnl,
        "total_pnl": total_pnl,
        "delta": base_delta,
        "hedge_quantity": hedge_qty,
    }
