"""
Educational path-dependent Monte Carlo for Athena / Phoenix autocall structured notes.

Payoff conventions (interview-style, discrete observations only):
------------------------------------------------------------------
- **Phoenix (non-memory):** at each observation before autocall, if performance ≥ coupon barrier,
  pay one coupon quantum (digital-style). Missed periods are lost.
- **Phoenix memory:** missed coupon quanta accrue in a memory bank; when performance ≥ coupon barrier,
  pay bank + current quantum, then reset the bank to zero.
- **Athena:** no interim coupons. The only coupon-like cashflow on the coupon schedule is paid **together
  with autocall redemption** as *notional + one coupon quantum* when the autocall barrier is breached.
- **Autocall:** if performance ≥ autocall barrier on an observation date, the note redeems that date for
  **notional + one coupon quantum** (same quantum as Phoenix uses per period for consistency).
  On that date we do **not** also pay a separate Phoenix digital coupon (single redemption cheque).
- **Coupon quantum (user-selectable):** either *prorated by schedule* (annual % × time between observation
  dates × notional) or *full headline per date* (annual % × notional on every coupon and on the autocall coupon leg).
- **Final redemption** if the note survives to maturity: if final performance ≥ capital protection barrier,
  receive **100% notional**; otherwise receive **notional × final performance** (ratio vs initial reference).

Knock-in level is still tracked for risk metrics but does **not** gate the final redemption formula above
(teaching simplification — many live deals layer KI into downside; extend here if you want that linkage).

Diagnostics: *coupon_pv* aggregates all discounted coupon cashflows (Phoenix interim plus the coupon leg
of autocall redemption). *downside_pv* is the discounted maturity principal shortfall vs 100% notional on
paths that reach the final observation without autocall; it is not comparable to knock-in probability
(intraday KI does not reduce principal in this teaching model unless the final fixing is below the capital barrier).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

Frequency = Literal["monthly", "quarterly", "semi_annual", "annual"]
CouponConvention = Literal["prorated_by_schedule", "full_headline_per_date"]
FamilyKey = Literal[
    "athena_autocall",
    "phoenix_autocall_memory",
    "phoenix_autocall_no_memory",
]


def investor_discount(rate: float, issuer_spread: float, t_years: float) -> float:
    return math.exp(-(rate + max(issuer_spread, 0.0)) * t_years)


def skew_sigma(px: np.ndarray, ref_level: float, atm_vol: float, skew_steepness: float) -> np.ndarray:
    """Local vol multiplier: lower spot => higher sigma (equity skew / crash premium).

    Mirrors the idea that OTM puts trade rich: as spot falls, effective vol rises. This is *not* a full
    stochastic-vol model; spot–vol correlation (leverage effect) is only sketched via this path-dependent
    vol map — use the separate spot–vol rho input in the UI as a qualitative overlay.
    """
    rel = ref_level / np.maximum(px, 1e-12)
    stress = np.maximum(0.0, rel * rel - 1.0)
    sig = atm_vol * (1.0 + skew_steepness * stress)
    return np.clip(sig, 0.08, float(max(atm_vol * 3.0, atm_vol)))


def simulate_single(paths: int, s0: float, T: float, steps: int, r: float, q: float, atm: float, skew: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = T / float(steps)
    s = np.zeros((paths, steps + 1))
    s[:, 0] = s0
    sqrt_dt = math.sqrt(dt)
    for j in range(steps):
        sig = skew_sigma(s[:, j], s0, atm, skew)
        drift = (r - q - 0.5 * sig * sig) * dt
        z = rng.standard_normal(paths)
        s[:, j + 1] = s[:, j] * np.exp(drift + sig * sqrt_dt * z)
    return s


def pairwise_corr_matrix(n_assets: int, rho: float) -> np.ndarray:
    m = np.full((n_assets, n_assets), float(rho))
    np.fill_diagonal(m, 1.0)
    return m


def simulate_multi(paths: int, spots: np.ndarray, T: float, steps: int, r: float, q: float, atm: float, skew: float, rho: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_assets = spots.shape[0]
    L = np.linalg.cholesky(pairwise_corr_matrix(n_assets, rho))
    dt = T / float(steps)
    s = np.zeros((paths, n_assets, steps + 1))
    s[:, :, 0] = spots
    sqrt_dt = math.sqrt(dt)

    for st in range(steps):
        z = rng.standard_normal((paths, n_assets)) @ L.T
        drift = np.zeros_like(z)
        diff = np.zeros_like(z)
        for k in range(n_assets):
            ref = spots[k]
            sig = skew_sigma(s[:, k, st], ref, atm, skew)
            drift[:, k] = (r - q - 0.5 * sig * sig) * dt
            diff[:, k] = sig * sqrt_dt * z[:, k]
        s[:, :, st + 1] = s[:, :, st] * np.exp(drift + diff)

    return s


def observation_steps(T: float, steps: int, freq: Frequency) -> np.ndarray:
    per_year = {"monthly": 12, "quarterly": 4, "semi_annual": 2, "annual": 1}[freq]
    dt = T / float(steps)
    n_dates = max(2, math.ceil(per_year * T))
    ix = []
    for j in range(n_dates):
        t = T * (j + 1) / n_dates
        i = min(steps - 1, max(0, int(round(t / dt)) - 1))
        ix.append(i)
    ix = sorted(set(ix))
    if ix[-1] != steps - 1:
        ix.append(steps - 1)
    return np.array(ix, dtype=int)


def coupon_cash_per_visit(
    T: float,
    annual_rate: float,
    notional: float,
    n_dates: int,
    convention: CouponConvention,
) -> float:
    """Cash coupon quantum for one Phoenix payment or for the coupon leg of an autocall redemption."""
    if n_dates <= 1:
        return annual_rate * T * notional
    if convention == "prorated_by_schedule":
        dtau = T / float(max(n_dates - 1, 1))
        return annual_rate * dtau * notional
    return annual_rate * notional


def coupon_convention_summary(convention: CouponConvention, T: float, annual_rate: float, notional: float, n_dates: int) -> str:
    cash = coupon_cash_per_visit(T, annual_rate, notional, n_dates, convention)
    if convention == "prorated_by_schedule":
        if n_dates <= 1:
            line = f"Prorated: single observation pays annual % x maturity x notional = {cash:,.2f}."
        else:
            dtau = T / float(max(n_dates - 1, 1))
            line = (
                f"Prorated: each coupon uses dt = T/(N_obs-1) = {dtau:.4f} yrs; "
                f"cash = annual % x dt x notional = {cash:,.2f} per date."
            )
    else:
        line = (
            f"Full headline: each coupon / autocall coupon leg pays annual % x notional = {cash:,.2f} "
            "(full headline amount each event, not spread over the schedule)."
        )
    return line


def perf_level(paths: np.ndarray, refs, step_idx: int, worst_of: bool) -> np.ndarray:
    if worst_of:
        r = np.maximum(np.asarray(refs, dtype=float), 1e-12)
        return np.min(paths[:, :, step_idx] / r[np.newaxis, :], axis=1)
    return paths[:, step_idx] / max(float(refs), 1e-12)


@dataclass
class StructuredPricingInput:
    notional: float
    spot: float
    initial_level: float
    strike: float
    maturity_years: float
    risk_free_rate: float
    dividend_yield: float
    issuer_spread: float
    atm_volatility: float
    skew_steepness: float
    correlation: float

    autocall_barrier_ratio: float
    coupon_barrier_ratio: float
    knock_in_barrier_ratio: float
    capital_recovery_barrier_ratio: float

    coupon_level_annual_pct: float
    coupon_convention: CouponConvention
    observation_frequency: Frequency
    family: FamilyKey

    use_worst_of: bool
    n_assets: int

    n_simulations: int
    n_time_steps: int
    issue_price_pct: float
    redemption_floor_ratio: float = 0.0


def price_structured_product(inp: StructuredPricingInput, seed: int = 481_51623) -> dict[str, Any]:
    """Path-dependent PV under Q; cashflows discounted at (r + issuer_spread)."""
    # Athena: coupon is paid only on autocall redemption — same performance trigger (single barrier).
    if inp.family == "athena_autocall":
        inp = replace(inp, coupon_barrier_ratio=float(inp.autocall_barrier_ratio))
    n_paths = inp.n_simulations
    steps = inp.n_time_steps
    obs_ix = observation_steps(inp.maturity_years, steps, inp.observation_frequency)

    pv_coupon_vec = np.zeros(n_paths)
    pv_red_early_vec = np.zeros(n_paths)
    pv_red_early_coupon_vec = np.zeros(n_paths)
    pv_red_early_notional_vec = np.zeros(n_paths)
    pv_red_terminal_vec = np.zeros(n_paths)

    cash_visit = coupon_cash_per_visit(
        inp.maturity_years,
        inp.coupon_level_annual_pct,
        inp.notional,
        len(obs_ix),
        inp.coupon_convention,
    )

    worst_of = inp.use_worst_of
    if worst_of:
        n_a = max(2, inp.n_assets)
        spot_vec = np.full(n_a, float(inp.spot))
        refs = spot_vec.copy()
        paths_all = simulate_multi(n_paths, spot_vec, inp.maturity_years, steps, inp.risk_free_rate, inp.dividend_yield,
                                   inp.atm_volatility, inp.skew_steepness, inp.correlation, seed)
    else:
        ref = float(inp.initial_level) if inp.initial_level > 0 else float(inp.spot)
        paths_all = simulate_single(n_paths, float(inp.spot), inp.maturity_years, steps, inp.risk_free_rate,
                                    inp.dividend_yield, inp.atm_volatility, inp.skew_steepness, seed)
        refs = ref

    autocall = np.zeros(n_paths, dtype=bool)
    ki = np.zeros(n_paths, dtype=bool)

    memory = np.zeros(n_paths)
    coupon_ok_count = np.zeros(n_paths)
    phoenix_live_dates = np.zeros(n_paths)

    for seq, ix in enumerate(obs_ix):
        ix_int = int(ix)
        lvl = perf_level(paths_all, refs, ix_int, worst_of)
        t_frac = inp.maturity_years * (ix_int + 1) / float(steps)
        df = investor_discount(inp.risk_free_rate, inp.issuer_spread, t_frac)

        ki |= lvl <= inp.knock_in_barrier_ratio

        eligible = (~autocall.astype(bool))

        # --- Autocall first (same date: one cheque notional + coupon; no double Phoenix coupon) ---
        trig = eligible & (lvl.astype(float) >= inp.autocall_barrier_ratio)
        early_n = trig.astype(float) * inp.notional
        early_c = trig.astype(float) * cash_visit
        pv_red_early_notional_vec += df * early_n
        pv_red_early_coupon_vec += df * early_c
        pv_red_early_vec += df * (early_n + early_c)
        autocall |= trig.astype(bool)

        coupon_eligible = eligible & (~trig.astype(bool))

        if inp.family in ("phoenix_autocall_memory", "phoenix_autocall_no_memory"):
            phoenix_live_dates += coupon_eligible.astype(float)

        if inp.family == "phoenix_autocall_memory":
            good_obs = coupon_eligible & (lvl.astype(float) >= inp.coupon_barrier_ratio)
            bad_obs = coupon_eligible & (lvl.astype(float) < inp.coupon_barrier_ratio)

            pay_coupon_now = np.zeros(n_paths)
            pay_coupon_now[good_obs] = (memory.astype(float))[good_obs] + cash_visit
            coupon_ok_count += good_obs.astype(float)
            pv_coupon_vec += df * pay_coupon_now

            memory = np.where(good_obs.astype(bool), 0.0, memory)
            memory = memory + bad_obs.astype(float) * cash_visit

        elif inp.family == "phoenix_autocall_no_memory":
            good_obs = coupon_eligible & (lvl.astype(float) >= inp.coupon_barrier_ratio)
            pay_coupon_now = np.zeros(n_paths)
            pay_coupon_now[good_obs] = cash_visit
            coupon_ok_count += good_obs.astype(float)
            pv_coupon_vec += df * pay_coupon_now

        # Athena: interim coupons intentionally suppressed (coupon only via autocall redemption above).

        if np.all(autocall):
            break

    idx_terminal = obs_ix[-1]
    lvl_T = perf_level(paths_all, refs, int(idx_terminal), worst_of)

    # Final redemption if no autocall: full notional if final ≥ capital protection barrier, else notional × perf.
    terminal_cash = np.zeros(n_paths)
    alive = ~autocall.astype(bool)
    full_nominal = alive & (lvl_T.astype(float) >= inp.capital_recovery_barrier_ratio)
    scaled_nominal = alive & (lvl_T.astype(float) < inp.capital_recovery_barrier_ratio)

    # Below capital barrier: notional * max(performance, floor) — floor caps worst participation (PDI-style).
    floor_r = float(np.clip(inp.redemption_floor_ratio, 0.0, 1.0))
    perf_eff = np.maximum(lvl_T.astype(float), floor_r)
    terminal_cash[full_nominal] = inp.notional
    terminal_cash[scaled_nominal] = (inp.notional * perf_eff)[scaled_nominal]

    df_T = investor_discount(inp.risk_free_rate, inp.issuer_spread, inp.maturity_years)
    pv_red_terminal_vec += alive.astype(float) * df_T * terminal_cash

    # Per path: autocall -> discounted early (notional + coupon); else -> terminal only. No overlap.
    pv_total = pv_coupon_vec + pv_red_early_vec + pv_red_terminal_vec

    fv_mean = float(np.mean(pv_total))
    fv_se = float(np.std(pv_total, ddof=1) / math.sqrt(max(n_paths - 1, 1)))

    # Maturity-only principal shortfall (alive paths). Autocall paths: terminal_cash stays 0 but are
    # excluded via `alive`; no double-count with pv_red_early (terminal PV is zero when autocall).
    terminal_principal_shortfall = np.zeros(n_paths)
    terminal_principal_shortfall[scaled_nominal] = (
        inp.notional - terminal_cash.astype(float)
    )[scaled_nominal]
    put_like_pv_mean = float(np.mean(terminal_principal_shortfall[alive.astype(bool)]) * df_T)

    zcb_nominal_pv = inp.notional * df_T

    prob_autocall = float(np.mean(autocall))
    prob_ki = float(np.mean(ki.astype(bool)))
    prob_loss = float(
        np.mean(
            alive.astype(bool)
            & (lvl_T.astype(float) < inp.capital_recovery_barrier_ratio)
        )
    )
    if inp.family == "athena_autocall":
        prob_all_coupons_ph = float("nan")
    else:
        # P(coupon paid on every observation where the note was alive and autocall did not trigger that day).
        has_live = phoenix_live_dates > 0
        all_hit = coupon_ok_count >= phoenix_live_dates
        prob_all_coupons_ph = float(np.mean(all_hit[has_live])) if np.any(has_live) else float("nan")

    bucket_edges = np.linspace(0.5, 1.35, 22)
    mid_pts, vals = [], []
    for low, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask_level = (lvl_T >= low) & (lvl_T < hi)
        mid_pts.append(0.5 * (low + hi))
        vals.append(float(np.nanmean(pv_total[mask_level])) if mask_level.any() else math.nan)

    loss_face = alive.astype(bool) & (lvl_T.astype(float) < inp.capital_recovery_barrier_ratio)
    expected_loss_conditional = (
        float(np.mean(terminal_principal_shortfall[loss_face])) if np.any(loss_face) else float("nan")
    )

    redemption_scenarios = dict(
        autocall_early=float(np.mean(autocall)),
        mature_full_cap=float(np.mean(~autocall.astype(bool) & (terminal_cash >= inp.notional * 0.999))),
        mature_loss=float(np.mean(~autocall.astype(bool) & (terminal_cash < inp.notional * 0.999))),
    )

    pv_coupon_all_paths = pv_coupon_vec + pv_red_early_coupon_vec

    return dict(
        pv_total_paths=pv_total,
        pv_coupon_paths=pv_coupon_vec,
        pv_autocall_coupon_paths=pv_red_early_coupon_vec,
        pv_early_red_paths=pv_red_early_vec,
        pv_early_red_notional_paths=pv_red_early_notional_vec,
        pv_terminal_red_paths=pv_red_terminal_vec,
        terminal_payoff_lvl=terminal_cash,
        final_perf=lvl_T,
        autocall_hit=autocall,
        knock_in_hit=ki.astype(bool),
        coupon_convention=inp.coupon_convention,
        coupon_cash_per_visit=float(cash_visit),
        coupon_convention_detail=coupon_convention_summary(
            inp.coupon_convention,
            inp.maturity_years,
            inp.coupon_level_annual_pct,
            inp.notional,
            len(obs_ix),
        ),
        fair_mean=fv_mean,
        fair_se=fv_se,
        fair_pct_of_notional=100 * fv_mean / max(inp.notional, 1e-12),
        issue_price_pv=inp.notional * inp.issue_price_pct / 100,
        issue_price_pct=inp.issue_price_pct,
        prob_autocall=prob_autocall,
        prob_ki=prob_ki,
        prob_capital_loss=prob_loss,
        prob_all_coupons=prob_all_coupons_ph,
        exp_coupon_pv=float(np.mean(pv_coupon_all_paths)),
        zcb_pv_nominal=zcb_nominal_pv,
        coupon_pv=float(np.mean(pv_coupon_all_paths)),
        coupon_pv_interim=float(np.mean(pv_coupon_vec)),
        coupon_pv_autocall=float(np.mean(pv_red_early_coupon_vec)),
        downside_pv=put_like_pv_mean,
        exp_maturity_approx=prob_autocall * 0.62 * inp.maturity_years + (1 - prob_autocall) * inp.maturity_years,
        payoff_bucket_mid=mid_pts,
        payoff_bucket_pv=np.asarray(vals),
        expected_loss_given_cap_breach=expected_loss_conditional,
        redemption_scenarios_prob=redemption_scenarios,
        expected_discounted_payoff=fv_mean,
    )


def sensitivity_table(inp: StructuredPricingInput) -> dict[str, float]:
    sens_seed = 100_003
    if inp.family == "athena_autocall":
        inp = replace(inp, coupon_barrier_ratio=float(inp.autocall_barrier_ratio))
    base_line = price_structured_product(inp, seed=sens_seed)

    def reprice(updated: StructuredPricingInput) -> float:
        # Same seed across bumps = common random numbers, stable finite differences.
        return price_structured_product(updated, seed=sens_seed)["fair_mean"]

    def clone(**kw) -> StructuredPricingInput:
        d = inp.__dict__.copy()
        d.update(kw)
        return StructuredPricingInput(**d)

    ds = max(inp.spot * 0.01, 0.25)

    deltas = {}

    deltas["fair_value_pv_per_$spot"] = (reprice(clone(spot=inp.spot + ds)) - base_line["fair_mean"]) / ds

    dv = max(inp.atm_volatility * 0.06, 0.005)
    deltas["fair_value_pv_per_1vol_pt"] = (reprice(clone(atm_volatility=inp.atm_volatility + dv)) - base_line["fair_mean"]) / (dv * 100.0)

    dskew = 0.05
    deltas["fair_value_pv_per_skew_unit"] = (reprice(clone(skew_steepness=inp.skew_steepness + dskew)) - base_line["fair_mean"]) / dskew

    dr = 0.0025
    deltas["fair_value_pv_per_bp_rate"] = (reprice(clone(risk_free_rate=inp.risk_free_rate + dr)) - base_line["fair_mean"]) / (dr * 10_000.0)

    dq = 0.0025
    deltas["fair_value_pv_per_bp_div"] = (reprice(clone(dividend_yield=inp.dividend_yield + dq)) - base_line["fair_mean"]) / (dq * 10_000.0)

    db = 0.005

    deltas["fair_value_pv_per_ki_barrier"] = (reprice(clone(knock_in_barrier_ratio=inp.knock_in_barrier_ratio - db)) - base_line["fair_mean"]) / db

    deltas["fair_value_pv_per_autocall_barrier"] = (
        reprice(clone(autocall_barrier_ratio=max(inp.autocall_barrier_ratio - db, 1e-4))) - base_line["fair_mean"]
    ) / db

    if inp.family == "athena_autocall":
        # Same barrier as autocall; isolated coupon-barrier bump is not a meaningful Athena degree of freedom.
        deltas["fair_value_pv_per_coupon_barrier"] = float("nan")
    else:
        deltas["fair_value_pv_per_coupon_barrier"] = (
            reprice(clone(coupon_barrier_ratio=max(inp.coupon_barrier_ratio - db, 1e-4))) - base_line["fair_mean"]
        ) / db

    if inp.use_worst_of:
        drho = min(0.05, max(0.01, abs(inp.correlation) * 0.1))
        deltas["fair_value_pv_per_corr"] = (reprice(clone(correlation=min(inp.correlation + drho, 0.99))) - base_line["fair_mean"]) / drho

    deltas["fair_value_pv_per_cap_barrier"] = (
        reprice(clone(capital_recovery_barrier_ratio=max(inp.capital_recovery_barrier_ratio - db, 1e-4)))
        - base_line["fair_mean"]
    ) / db

    return deltas


def structuring_analysis(inp: StructuredPricingInput, fair_mean: float, target_issue_pct: float) -> dict[str, Any]:
    """Educational decomposition: ZCB(PV of notional bullet at T) + option package ≈ fair value.

    Formulas (same discount as cashflows, investor curve r + s):
      df_T = exp(-(r + s) * T)
      zcb_pv = N * df_T
      target_pv = N * (target_issue_pct / 100)
      structuring_budget = target_pv - zcb_pv
      option_package_pv = fair_mean - zcb_pv
      gap_to_target = target_pv - fair_mean

    At par: fair_mean ≈ target_pv ⇒ option_package_pv ≈ structuring_budget (MC noise aside).
    """
    df_T = investor_discount(inp.risk_free_rate, inp.issuer_spread, inp.maturity_years)
    zcb_pv = inp.notional * df_T
    target_pv = inp.notional * (target_issue_pct / 100.0)
    structuring_budget = target_pv - zcb_pv
    option_package_pv = fair_mean - zcb_pv
    gap_abs = target_pv - fair_mean
    gap_pct_pts = 100.0 * gap_abs / max(inp.notional, 1e-12)
    tol = max(1e-9 * inp.notional, 1.0)
    option_fits_budget = option_package_pv <= structuring_budget + tol
    util = option_package_pv / max(structuring_budget, 1e-9)

    return dict(
        zero_coupon_pv=zcb_pv,
        zero_coupon_pct_of_notional=100.0 * zcb_pv / max(inp.notional, 1e-12),
        target_issue_pct=target_issue_pct,
        target_pv=target_pv,
        structuring_budget_pv=structuring_budget,
        structuring_budget_pct_of_notional=100.0 * structuring_budget / max(inp.notional, 1e-12),
        option_package_pv=option_package_pv,
        option_package_pct_of_notional=100.0 * option_package_pv / max(inp.notional, 1e-12),
        gap_to_target_abs=gap_abs,
        gap_to_target_pct_pts=gap_pct_pts,
        option_fits_budget=bool(option_fits_budget),
        budget_utilization_pct=100.0 * float(util),
    )


def calibrate_coupon_for_target(
    base: StructuredPricingInput,
    target_fair_pv: float,
    seed: int = 77_777,
    tol_abs: float | None = None,
    max_iter: int = 48,
) -> tuple[float, dict[str, Any]]:
    """Bisection on annual coupon rate so that fair PV ≈ target (monotone in coupon for this model)."""
    tol = tol_abs if tol_abs is not None else max(5e-4 * base.notional, 25.0)
    lo, hi = 0.0, 0.55

    def fv_at(c: float) -> float:
        return price_structured_product(replace(base, coupon_level_annual_pct=float(c)), seed=seed)["fair_mean"]

    fv_lo = fv_at(lo)
    fv_hi = fv_at(hi)
    meta: dict[str, Any] = {"fv_lo": fv_lo, "fv_hi": fv_hi}

    if target_fair_pv <= fv_lo:
        return lo, {**meta, "status": "at_lower_bound", "fair": fv_lo}
    expand = 0
    while target_fair_pv > fv_hi and hi < 2.5 and expand < 12:
        hi = min(2.5, hi * 1.35 + 0.05)
        fv_hi = fv_at(hi)
        expand += 1
    meta["fv_hi"] = fv_hi
    meta["hi_after_expand"] = hi

    if target_fair_pv > fv_hi:
        return hi, {**meta, "status": "above_bracket", "fair": fv_hi}

    for k in range(max_iter):
        mid = 0.5 * (lo + hi)
        fv_mid = fv_at(mid)
        if abs(fv_mid - target_fair_pv) < tol:
            return mid, {**meta, "status": "ok", "fair": fv_mid, "iterations": k}
        if fv_mid < target_fair_pv:
            lo = mid
        else:
            hi = mid

    mid = 0.5 * (lo + hi)
    fv_mid = fv_at(mid)
    return mid, {**meta, "status": "max_iter", "fair": fv_mid, "iterations": max_iter}


def sweep_coupon_fair_values(inp: StructuredPricingInput, coupon_rates: list[float], seed: int) -> tuple[list[float], list[float]]:
    """Fair PV vs coupon grid (same seed = smooth curve)."""
    fairs: list[float] = []
    for c in coupon_rates:
        fairs.append(price_structured_product(replace(inp, coupon_level_annual_pct=float(c)), seed=seed)["fair_mean"])
    return coupon_rates, fairs


def sweep_autocall_barrier_fair_values(inp: StructuredPricingInput, barriers: list[float], seed: int) -> tuple[list[float], list[float]]:
    fairs: list[float] = []
    for b in barriers:
        fairs.append(
            price_structured_product(replace(inp, autocall_barrier_ratio=float(np.clip(b, 1e-4, 2.5))), seed=seed)[
                "fair_mean"
            ]
        )
    return barriers, fairs


def sweep_correlation_fair_values(inp: StructuredPricingInput, rhos: list[float], seed: int) -> tuple[list[float], list[float]]:
    fairs: list[float] = []
    for rho in rhos:
        fairs.append(
            price_structured_product(replace(inp, correlation=float(np.clip(rho, 0.01, 0.99))), seed=seed)["fair_mean"]
        )
    return rhos, fairs


def monte_carlo_greeks(inp: StructuredPricingInput, seed: int = 223_558) -> dict[str, float]:
    # One RNG stream for all bumps (common random numbers); otherwise delta is drowned by independent MC noise.
    base = price_structured_product(inp, seed=seed)
    bump_spot = max(inp.spot * 0.01, 0.2)
    pv_up = price_structured_product(StructuredPricingInput(**{**inp.__dict__, **{"spot": inp.spot + bump_spot}}), seed=seed)
    pv_dn = price_structured_product(StructuredPricingInput(**{**inp.__dict__, **{"spot": inp.spot - bump_spot}}), seed=seed)

    denom = max(inp.notional, 1e-12)

    delta = (pv_up["fair_mean"] - pv_dn["fair_mean"]) / (2 * bump_spot) / denom
    gamma = (pv_up["fair_mean"] - 2 * base["fair_mean"] + pv_dn["fair_mean"]) / (bump_spot * bump_spot) / denom

    dv = max(inp.atm_volatility * 0.06, 0.005)
    v_up = price_structured_product(StructuredPricingInput(**{**inp.__dict__, **{"atm_volatility": inp.atm_volatility + dv}}), seed=seed)
    vega = (v_up["fair_mean"] - base["fair_mean"]) / dv / denom

    dr = 0.0025
    r_up = price_structured_product(StructuredPricingInput(**{**inp.__dict__, **{"risk_free_rate": inp.risk_free_rate + dr}}), seed=seed)
    rho = (r_up["fair_mean"] - base["fair_mean"]) / dr / denom

    return dict(
        delta_pv_per_pct_notional=100 * delta,
        gamma_pv_scaled=gamma,
        vega_per_vol_pt=100 * vega,
        rho_per_rate_unit=rho,
    )

def payoff_explanation(family: FamilyKey) -> str:
    atlas = (
        "**Athena:** no running coupons. If the autocall test passes on an observation date, you receive "
        "**notional + one coupon quantum** that day and the note ends. Otherwise you wait for the final "
        "redemption rule (100% notional if final performance is at/above the capital protection barrier, "
        "else notional × final performance vs the initial reference)."
    )
    pm = (
        "**Phoenix memory:** on each live observation, if performance is above the **coupon barrier**, you are paid "
        "any **missed coupons plus the current coupon**; if you miss, coupons **accrue** until regained. "
        "If the **autocall** barrier is hit first on a date, you receive **notional + coupon** that day and "
        "the running coupon stream stops (no double coupon on the autocall cheque)."
    )
    pn = (
        "**Phoenix (non-memory):** each live observation above the **coupon barrier** pays **only** the current "
        "coupon quantum; missed periods are **lost**. Autocall still pays **notional + coupon** and terminates."
    )

    lookup = dict(
        athena_autocall=atlas,
        phoenix_autocall_memory=pm,
        phoenix_autocall_no_memory=pn,
    )
    return lookup[family]
