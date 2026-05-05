"""
Educational structuring-budget helpers: ZC funding leg, isolated PDI-style downside MC,
and digital leg metrics sourced from the full structured MC engine.

Skew: reuses skew_sigma from structured_products (higher vol when spot is lower).
Drift: r - q - 0.5*sigma^2 per step in path simulation.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from core.structured_products import (
    Frequency,
    StructuredPricingInput,
    investor_discount,
    observation_steps,
    price_structured_product,
    simulate_single,
)

KIStyle = Literal["discrete", "continuous"]


def funding_leg(
    notional: float,
    maturity_years: float,
    risk_free_rate: float,
    issuer_spread: float,
    target_issue_pct: float,
) -> dict[str, float]:
    """
    Bullet zero-coupon on notional at T discounted at investor curve (r + funding spread).

    df_T = exp(-(r + s) * T); zc_pv = N * df_T. Higher r or s => smaller df => lower ZC PV =>
    larger (target_pv - zc_pv) option budget for the same target issue price.
    """
    df_t = investor_discount(risk_free_rate, issuer_spread, maturity_years)
    zc_pv = notional * df_t
    zc_pct = 100.0 * df_t
    target_pv = notional * (target_issue_pct / 100.0)
    option_budget = target_pv - zc_pv
    return dict(
        df_T=df_t,
        zc_pv=zc_pv,
        zc_pct_of_notional=zc_pct,
        target_issue_pct=target_issue_pct,
        target_pv=target_pv,
        option_budget_from_zc=option_budget,
        option_budget_pct_of_notional=100.0 * option_budget / max(notional, 1e-12),
    )


def mcp_pdi_slice(
    *,
    notional: float,
    initial_level: float,
    maturity_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    issuer_spread: float,
    atm_vol: float,
    skew_steepness: float,
    ki_barrier_ratio: float,
    pdi_strike_ratio: float,
    n_paths: int,
    n_steps: int,
    frequency: Frequency,
    ki_style: KIStyle,
    seed: int,
) -> dict[str, Any]:
    """
    Single-underlying path-dependent slice for knock-in + terminal participation.

    - KI: discrete = barrier checked on observation schedule; continuous = min performance
      over all time steps vs initial level.
    Classic Put Down-and-In (PDI), investor long / issuer short:
      if KI never touched -> payoff = 0
      if KI touched       -> payoff = max(K - S_T, 0), K = pdi_strike_ratio * initial_level

    premium_short_pdi is the discounted expected payoff (positive) that the issuer receives for
    selling this downside option; in the structuring budget it increases available option budget.
    """
    ref = max(float(initial_level), 1e-12)
    strike_abs = float(pdi_strike_ratio) * ref
    paths = simulate_single(
        n_paths,
        float(initial_level),
        maturity_years,
        n_steps,
        risk_free_rate,
        dividend_yield,
        atm_vol,
        skew_steepness,
        seed,
    )
    perf = paths / ref
    if ki_style == "continuous":
        ki = np.min(perf, axis=1) <= ki_barrier_ratio
    else:
        ki = np.zeros(n_paths, dtype=bool)
        for ix in observation_steps(maturity_years, n_steps, frequency):
            i = int(ix)
            ki |= perf[:, i] <= ki_barrier_ratio

    s_t = paths[:, -1].astype(float)
    payoff = np.zeros(n_paths)
    hit = ki.astype(bool)
    payoff[hit] = np.maximum(strike_abs - s_t[hit], 0.0) * (notional / ref)

    df_t = investor_discount(risk_free_rate, issuer_spread, maturity_years)
    pv_paths = payoff * df_t
    premium_short_pdi = float(np.mean(pv_paths))

    prob_ki = float(np.mean(ki))
    mask = ki.astype(bool)
    cond = float(np.mean(payoff[mask])) if np.any(mask) else float("nan")

    return dict(
        premium_short_pdi=premium_short_pdi,
        pdi_pv_investor_loss=premium_short_pdi,
        prob_ki=prob_ki,
        expected_loss_abs_given_ki=cond,
        pv_loss_paths=pv_paths,
    )


def digital_leg_from_note(inp: StructuredPricingInput, seed: int) -> dict[str, Any]:
    """Coupon + autocall coupon leg PV from the full path-dependent engine (transparent reuse)."""
    v = price_structured_product(inp, seed=seed)
    return dict(
        digital_pv=float(v["coupon_pv"]),
        digital_pv_pct_notional=100.0 * float(v["coupon_pv"]) / max(inp.notional, 1e-12),
        coupon_cash_per_visit=float(v["coupon_cash_per_visit"]),
        prob_autocall=float(v["prob_autocall"]),
        prob_all_coupons=v.get("prob_all_coupons"),
        fair_mean=float(v["fair_mean"]),
    )


def structuring_waterfall(
    funding: dict[str, float],
    pdi: dict[str, Any],
    digital: dict[str, Any],
    fair_mc: float,
    notional: float,
) -> dict[str, Any]:
    """
    Budget walk (currency, same notional scale):

      option_budget_0 = target_pv - zc_pv
      after_pdi     = option_budget_0 + premium_short_pdi
      remaining     = after_pdi - digital_pv
      net_option    = -premium_short_pdi + digital_pv   (issuer net cost of options vs bullet)

    Reconciliation (approximate): zc + net_option ≈ fair only when components are weakly coupled.
    """
    zc = float(funding["zc_pv"])
    tgt = float(funding["target_pv"])
    b0 = float(funding["option_budget_from_zc"])
    pdi_prem = float(pdi["premium_short_pdi"])
    dig = float(digital["digital_pv"])
    after_pdi = b0 + pdi_prem
    remaining = after_pdi - dig
    net_option = dig - pdi_prem
    gap_abs = tgt - fair_mc
    gap_pts = 100.0 * gap_abs / max(notional, 1e-12)
    recon = zc + net_option
    recon_gap = fair_mc - recon
    return dict(
        option_budget_initial=b0,
        after_short_pdi_premium=after_pdi,
        remaining_budget=remaining,
        net_option_package_pv=net_option,
        fair_mc=fair_mc,
        gap_to_target_abs=gap_abs,
        gap_to_target_pct_pts=gap_pts,
        reconciliation_zc_plus_net_option=recon,
        reconciliation_gap=recon_gap,
    )
