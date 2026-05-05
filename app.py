"""Streamlit app for an educational structured options pricer."""

from __future__ import annotations

import html
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.barrier_products import (
    price_barrier_option_mc,
    product_payoff_explanation,
    vanilla_black_scholes,
)
from core.structured_products import (
    StructuredPricingInput,
    monte_carlo_greeks,
    payoff_explanation,
    price_structured_product,
    sensitivity_table,
)
from core.structuring_budget import (
    digital_leg_from_note,
    funding_leg,
    mcp_pdi_slice,
    structuring_waterfall,
)
from core.black_scholes import call_price, put_price
from core.hedging import hedge_quantity, option_delta, simulate_delta_hedge_pnl
from core.monte_carlo import european_option_monte_carlo
from data.market_data import fetch_historical_volatility, fetch_market_option_data, fetch_synthetic_option_data
from data.news import fetch_general_market_news, fetch_major_indices
from instruments.volatility_strategies import straddle_price, strangle_price
from risk.greeks import call_delta, call_rho, call_theta, gamma, put_delta, put_rho, put_theta, vega


@st.cache_data(ttl=300, show_spinner=False)
def _get_market_option_data_cached(ticker: str, expiration: str, option_type: str):
    """Cached market data fetch (5 minutes TTL)."""
    return fetch_market_option_data(ticker=ticker, expiration=expiration, option_type=option_type)


@st.cache_data(ttl=600, show_spinner=False)
def _get_news_cached(limit: int = 8):
    """Cached market news fetch (10 minutes TTL)."""
    return fetch_general_market_news(limit=limit)


@st.cache_data(ttl=120, show_spinner=False)
def _get_major_indices_cached():
    """Cached major indices fetch (2 minutes TTL)."""
    return fetch_major_indices()


def _time_to_expiry_years(expiration: str) -> float:
    """Compute year fraction from today to expiration date."""
    expiry_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    today = datetime.now().date()
    days = (expiry_date - today).days
    # Keep strictly positive for Black-Scholes stability.
    return max(days / 365.0, 1.0 / 365.0)


def _clean_option_chain(option_chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Filter and enrich option chain for analytics."""
    chain = option_chain.copy()
    chain = chain[(chain["strike"] >= 0.7 * spot) & (chain["strike"] <= 1.3 * spot)]
    # For some expiries (especially very short-dated), Yahoo can have bid/ask at zero while lastPrice is populated.
    chain = chain[~((chain["bid"] <= 0) & (chain["ask"] <= 0) & (chain["lastPrice"] <= 0))]
    chain = chain[chain["impliedVolatility"].notna() & (chain["impliedVolatility"] > 0)]
    # Keep zero-volume strikes too: on some names/expiries they are still usable for smile diagnostics.
    chain = chain.sort_values("strike").reset_index(drop=True)

    mid_ba = (chain["bid"] + chain["ask"]) / 2.0
    chain["mid_price"] = np.where(mid_ba > 0, mid_ba, chain["lastPrice"])
    chain["spread_dollar"] = (chain["ask"] - chain["bid"]).clip(lower=0.0)
    chain["spread_pct"] = np.where(chain["mid_price"] > 0, (chain["spread_dollar"] / chain["mid_price"]) * 100.0, np.nan)
    chain["iv_pct"] = chain["impliedVolatility"] * 100.0
    return chain


def _quality_metrics(raw_chain: pd.DataFrame, clean_chain: pd.DataFrame, option_type: str) -> dict[str, float | int | str]:
    """Compute market quality diagnostics from cleaned option chain."""
    if clean_chain.empty:
        return {
            "raw_strikes": len(raw_chain),
            "filtered_strikes": 0,
            "retention_pct": 0.0,
            "median_spread_pct": 0.0,
            "median_spread_dollar": 0.0,
            "total_volume": 0,
            "total_open_interest": 0,
            "monotonicity_breaches": 0,
            "convexity_breaches": 0,
        }

    # Use strike-aware finite differences (strikes are not uniformly spaced).
    q = clean_chain.sort_values("strike")[["strike", "mid_price"]].dropna()
    q = q[(q["strike"] > 0) & (q["mid_price"] >= 0)]
    q = q.drop_duplicates(subset=["strike"], keep="first")
    strikes = q["strike"].to_numpy(dtype=float)
    prices = q["mid_price"].to_numpy(dtype=float)

    monotonicity_breaches = 0
    convexity_breaches = 0
    if len(prices) >= 2:
        # Relative tolerance avoids flagging tiny numerical / micro-quote noise as arbitrage.
        mono_tol = 1e-4 * float(np.nanmedian(np.maximum(prices, 1e-8)))
        first_diff = np.diff(prices)
        if option_type == "call":
            # Calls should be non-increasing with strike.
            monotonicity_breaches = int(np.sum(first_diff > mono_tol))
        else:
            # Puts should be non-decreasing with strike.
            monotonicity_breaches = int(np.sum(first_diff < -mono_tol))

    if len(prices) >= 3:
        # Butterfly convexity check with uneven strike spacing:
        # slope_i = dC/dK or dP/dK on segment i; convexity means slopes are non-decreasing in K.
        dk = np.diff(strikes)
        valid = dk > 1e-12
        slopes = np.diff(prices)[valid] / dk[valid]
        if len(slopes) >= 2:
            conv_tol = 1e-6
            convexity_breaches = int(np.sum(np.diff(slopes) < -conv_tol))

    return {
        "raw_strikes": int(len(raw_chain)),
        "filtered_strikes": int(len(clean_chain)),
        "retention_pct": float((len(clean_chain) / max(len(raw_chain), 1)) * 100.0),
        "median_spread_pct": float(clean_chain["spread_pct"].median(skipna=True)),
        "median_spread_dollar": float(clean_chain["spread_dollar"].median(skipna=True)),
        "total_volume": int(clean_chain["volume"].sum()),
        "total_open_interest": int(clean_chain["openInterest"].sum()),
        "monotonicity_breaches": monotonicity_breaches,
        "convexity_breaches": convexity_breaches,
    }


def _render_hedging_charts(pnl: dict, chart_view: str) -> None:
    """Render line and/or scenario bar chart from hedge simulation output."""
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=pnl["moves_pct"], y=pnl["option_pnl"], mode="lines", name="Option P&L"))
    line_fig.add_trace(go.Scatter(x=pnl["moves_pct"], y=pnl["hedge_pnl"], mode="lines", name="Hedge P&L"))
    line_fig.add_trace(go.Scatter(x=pnl["moves_pct"], y=pnl["total_pnl"], mode="lines", name="Total P&L"))
    line_fig.update_layout(
        title="Delta-Hedged P&L vs Spot Move",
        xaxis_title="Spot Move (%)",
        yaxis_title="P&L",
        template="plotly_white",
        legend_title="Component",
    )
    if chart_view in {"Both", "Line Only"}:
        st.plotly_chart(line_fig, use_container_width=True)

    scenario_points = [-10.0, 0.0, 10.0]
    scenario_indices = [0, len(pnl["moves_pct"]) // 2, len(pnl["moves_pct"]) - 1]
    option_vals = [float(pnl["option_pnl"][idx]) for idx in scenario_indices]
    hedge_vals = [float(pnl["hedge_pnl"][idx]) for idx in scenario_indices]
    total_vals = [float(pnl["total_pnl"][idx]) for idx in scenario_indices]

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(name="Option P&L", x=scenario_points, y=option_vals, text=[f"{v:.2f}" for v in option_vals]))
    bar_fig.add_trace(go.Bar(name="Hedge P&L", x=scenario_points, y=hedge_vals, text=[f"{v:.2f}" for v in hedge_vals]))
    bar_fig.add_trace(go.Bar(name="Total P&L", x=scenario_points, y=total_vals, text=[f"{v:.2f}" for v in total_vals]))
    bar_fig.update_layout(
        barmode="group",
        title="Delta-Hedged P&L at Key Spot Moves",
        xaxis_title="Spot Move (%)",
        yaxis_title="P&L",
        template="plotly_white",
        legend_title="Component",
    )
    if chart_view in {"Both", "Bar Only"}:
        st.markdown("**Key Scenarios (-10%, 0%, +10%)**")
        st.plotly_chart(bar_fig, use_container_width=True)


def render_simple_pricer() -> None:
    """Existing educational simple pricer."""
    st.caption("Clean educational app for Black-Scholes, Greeks, Monte Carlo, and volatility strategies.")

    st.markdown(
        """
        <style>
        .simple-pricer-section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .simple-pricer-section-subtitle {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Market Inputs")
    spot = st.sidebar.number_input("Spot Price (S)", min_value=0.01, value=100.0, step=1.0)
    strike = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
    rate = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.005, format="%.4f")
    volatility = st.sidebar.number_input("Volatility (sigma)", min_value=0.0001, value=0.20, step=0.01, format="%.4f")
    time_to_expiry = st.sidebar.number_input("Time to Expiry (T, years)", min_value=0.0001, value=1.0, step=0.1)

    st.sidebar.header("Monte Carlo Settings")
    n_simulations = st.sidebar.number_input("Simulations", min_value=1000, value=50000, step=1000)
    seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1)

    bs_call = call_price(spot, strike, rate, volatility, time_to_expiry)
    bs_put = put_price(spot, strike, rate, volatility, time_to_expiry)
    mc_call = european_option_monte_carlo(
        spot, strike, rate, volatility, time_to_expiry, n_simulations=n_simulations, option_type="call", seed=seed
    )
    mc_put = european_option_monte_carlo(
        spot, strike, rate, volatility, time_to_expiry, n_simulations=n_simulations, option_type="put", seed=seed
    )

    with st.container(border=True):
        st.markdown('<div class="simple-pricer-section-title">Pricing Snapshot</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="simple-pricer-section-subtitle">Black-Scholes and Monte Carlo estimates at current inputs</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Black-Scholes Prices")
            st.metric("Call Price", f"{bs_call:.4f}")
            st.metric("Put Price", f"{bs_put:.4f}")
        with col2:
            st.subheader("Monte Carlo Prices")
            st.metric("Call Price (MC)", f"{mc_call:.4f}")
            st.metric("Put Price (MC)", f"{mc_put:.4f}")

    st.markdown("")
    st.markdown("---")
    st.markdown("")

    with st.container(border=True):
        st.markdown('<div class="simple-pricer-section-title">Greeks</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="simple-pricer-section-subtitle">First-order sensitivities from Black-Scholes</div>',
            unsafe_allow_html=True,
        )
        greeks_data = {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Call": [
                call_delta(spot, strike, rate, volatility, time_to_expiry),
                gamma(spot, strike, rate, volatility, time_to_expiry),
                vega(spot, strike, rate, volatility, time_to_expiry),
                call_theta(spot, strike, rate, volatility, time_to_expiry),
                call_rho(spot, strike, rate, volatility, time_to_expiry),
            ],
            "Put": [
                put_delta(spot, strike, rate, volatility, time_to_expiry),
                gamma(spot, strike, rate, volatility, time_to_expiry),
                vega(spot, strike, rate, volatility, time_to_expiry),
                put_theta(spot, strike, rate, volatility, time_to_expiry),
                put_rho(spot, strike, rate, volatility, time_to_expiry),
            ],
        }
        st.dataframe(greeks_data, hide_index=True, use_container_width=True)

    st.markdown("")
    st.markdown("---")
    st.markdown("")

    with st.container(border=True):
        st.markdown('<div class="simple-pricer-section-title">Volatility Strategies</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="simple-pricer-section-subtitle">Premium comparison for long straddle and long strangle</div>',
            unsafe_allow_html=True,
        )
        strategy_col1, strategy_col2 = st.columns(2, gap="large")
        with strategy_col1:
            st.markdown("**Long Straddle** (same strike for call and put)")
            straddle = straddle_price(spot, strike, rate, volatility, time_to_expiry)
            st.metric("Straddle Premium", f"{straddle:.4f}")
        with strategy_col2:
            st.markdown("**Long Strangle** (OTM put + OTM call)")
            put_strike = st.number_input("Put Strike (for strangle)", min_value=0.01, value=95.0, step=1.0)
            call_strike = st.number_input("Call Strike (for strangle)", min_value=0.01, value=105.0, step=1.0)
            if put_strike < call_strike:
                strangle = strangle_price(spot, put_strike, call_strike, rate, volatility, time_to_expiry)
                st.metric("Strangle Premium", f"{strangle:.4f}")
            else:
                st.error("For a standard strangle, put strike must be less than call strike.")

    st.markdown("")
    st.markdown("---")
    st.markdown("")

    with st.container(border=True):
        st.markdown('<div class="simple-pricer-section-title">Delta Hedging</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="simple-pricer-section-subtitle">Hedge sizing and scenario P&L for delta-neutral management</div>',
            unsafe_allow_html=True,
        )
        hedge_col1, hedge_col2, hedge_col3 = st.columns(3, gap="large")
        with hedge_col1:
            hedge_option_type = st.selectbox("Option Type", options=["call", "put"], index=0)
        with hedge_col2:
            hedge_position_side = st.selectbox("Position Side", options=["long", "short"], index=0)
        with hedge_col3:
            hedge_position_size = st.number_input("Option Position Size", min_value=1.0, value=1.0, step=1.0)

        current_delta = option_delta(hedge_option_type, spot, strike, rate, volatility, time_to_expiry)
        current_hedge_qty = hedge_quantity(current_delta, hedge_position_size, hedge_position_side)

        summary_col1, summary_col2 = st.columns(2, gap="large")
        with summary_col1:
            st.metric("Option Delta", f"{current_delta:.4f}")
        with summary_col2:
            st.metric("Hedge Quantity (shares)", f"{current_hedge_qty:.4f}")

        chart_view = st.radio("Chart View", options=["Both", "Line Only", "Bar Only"], index=0, horizontal=True)
        pnl = simulate_delta_hedge_pnl(
            option_type=hedge_option_type,
            side=hedge_position_side,
            position=hedge_position_size,
            spot=spot,
            strike=strike,
            rate=rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
        )
        _render_hedging_charts(pnl, chart_view)


def render_market_vanilla_options() -> None:
    """Market-backed option analytics in a desk-style dashboard layout."""
    st.caption("Market Overview + Option Analytics")
    st.markdown(
        """
        <style>
        .market-section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .market-section-subtitle {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Market Vanilla Settings")
    data_mode = st.sidebar.selectbox("Data Mode", options=["live", "synthetic"], index=1)
    ticker = st.sidebar.selectbox("Company / Ticker", options=["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL"])
    option_type = st.sidebar.selectbox("Option Type", options=["call", "put"], index=0)
    rate = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.045, step=0.005, format="%.4f")
    refresh_market_data = st.sidebar.button("Refresh Market Data")
    hedge_side = st.sidebar.selectbox("Hedge Position Side", options=["long", "short"], index=0)
    hedge_size = st.sidebar.number_input("Hedge Option Position Size", min_value=1.0, value=1.0, step=1.0)
    chart_view = st.sidebar.radio("Hedge Chart View", options=["Both", "Line Only", "Bar Only"], index=0)
    st.sidebar.caption("Market data cache TTL: 5 minutes")

    if refresh_market_data:
        _get_market_option_data_cached.clear()
        st.sidebar.success("Market data cache cleared.")

    try:
        if data_mode == "synthetic":
            initial_data = fetch_synthetic_option_data(ticker=ticker, expiration="", option_type=option_type, rate=rate)
        else:
            initial_data = _get_market_option_data_cached(ticker=ticker, expiration="", option_type=option_type)
    except RuntimeError as exc:
        st.error(f"Unable to load market data. Please retry later. Details: {exc}")
        return
    expiration = st.sidebar.selectbox("Expiration", options=initial_data.expirations, index=0)

    try:
        if data_mode == "synthetic":
            data = fetch_synthetic_option_data(ticker=ticker, expiration=expiration, option_type=option_type, rate=rate)
            hv20 = max(0.05, 0.85 * data.atm_iv)
        else:
            data = _get_market_option_data_cached(ticker=ticker, expiration=expiration, option_type=option_type)
            hv20 = fetch_historical_volatility(ticker=ticker, window=20)
    except RuntimeError as exc:
        st.error(f"Unable to load market analytics. Details: {exc}")
        return

    fetch_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    spot = data.spot_price
    strike = data.atm_strike
    volatility = max(data.atm_iv, 0.0001)
    time_to_expiry = _time_to_expiry_years(data.expiration)
    days_to_expiry = max(int(round(time_to_expiry * 365)), 1)
    iv_hv_spread = volatility - hv20

    raw_chain = data.option_chain.copy()
    clean_chain = _clean_option_chain(raw_chain, spot=spot)
    auto_expiry_note = None
    if data_mode == "live" and clean_chain.empty and len(initial_data.expirations) > 1:
        for alt_exp in initial_data.expirations:
            if alt_exp == data.expiration:
                continue
            try:
                alt_data = _get_market_option_data_cached(ticker=ticker, expiration=alt_exp, option_type=option_type)
            except RuntimeError:
                continue
            alt_chain = _clean_option_chain(alt_data.option_chain.copy(), spot=alt_data.spot_price)
            if not alt_chain.empty:
                data = alt_data
                spot = data.spot_price
                strike = data.atm_strike
                volatility = max(data.atm_iv, 0.0001)
                time_to_expiry = _time_to_expiry_years(data.expiration)
                days_to_expiry = max(int(round(time_to_expiry * 365)), 1)
                iv_hv_spread = volatility - hv20
                raw_chain = data.option_chain.copy()
                clean_chain = alt_chain
                auto_expiry_note = f"Selected expiry had no usable quotes. Switched to nearest liquid expiry: {data.expiration}."
                break
    quality = _quality_metrics(raw_chain, clean_chain, option_type=option_type)

    bs_price = call_price(spot, strike, rate, volatility, time_to_expiry) if option_type == "call" else put_price(
        spot, strike, rate, volatility, time_to_expiry
    )
    mc_price = european_option_monte_carlo(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
        n_simulations=100_000,
        option_type=option_type,
        seed=123,
    )

    delta_val = option_delta(option_type, spot, strike, rate, volatility, time_to_expiry)
    gamma_val = gamma(spot, strike, rate, volatility, time_to_expiry)
    vega_val = vega(spot, strike, rate, volatility, time_to_expiry)
    theta_val = (
        call_theta(spot, strike, rate, volatility, time_to_expiry)
        if option_type == "call"
        else put_theta(spot, strike, rate, volatility, time_to_expiry)
    )
    rho_val = (
        call_rho(spot, strike, rate, volatility, time_to_expiry)
        if option_type == "call"
        else put_rho(spot, strike, rate, volatility, time_to_expiry)
    )
    hedge_qty = hedge_quantity(delta_val, hedge_size, hedge_side)

    with st.container(border=True):
        st.markdown('<div class="market-section-title">Market Header</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="market-section-subtitle">Live snapshot for selected ticker and expiry</div>',
            unsafe_allow_html=True,
        )
        if auto_expiry_note:
            st.info(auto_expiry_note)
        if data_mode == "synthetic":
            st.info("Synthetic mode: generated option chain and robust implied volatility surface.")
        h1 = st.columns(4)
        h1[0].metric("Expiration", data.expiration)
        h1[1].metric("Days to Expiry", f"{days_to_expiry:d}")
        h1[2].metric("Spot", f"{spot:,.2f}")
        h1[3].metric("ATM Strike", f"{strike:,.2f}")
        h2 = st.columns(4)
        h2[0].metric("Risk-Free Rate", f"{rate:.2%}")
        h2[1].metric("ATM IV", f"{volatility:.2%}")
        h2[2].metric("HV (20d)", f"{hv20:.2%}")
        h2[3].metric("IV - HV", f"{iv_hv_spread:+.2%}")
        mode_tag = "synthetic" if data_mode == "synthetic" else "live"
        st.caption(f"Quote timestamp: {fetch_timestamp} ({mode_tag})")

    st.markdown("")
    st.markdown("---")
    st.markdown("")

    with st.container(border=True):
        st.markdown('<div class="market-section-title">Market Quality</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="market-section-subtitle">Liquidity and no-arbitrage diagnostics on cleaned chain</div>',
            unsafe_allow_html=True,
        )
        qcols = st.columns(4)
        qcols[0].metric("Raw Strikes", f"{quality['raw_strikes']:,d}")
        qcols[1].metric("Filtered Strikes", f"{quality['filtered_strikes']:,d}")
        qcols[2].metric("Retention", f"{quality['retention_pct']:.1f}%")
        qcols[3].metric("Median Spread", f"{quality['median_spread_dollar']:.3f} $ / {quality['median_spread_pct']:.2f}%")
        qcols2 = st.columns(4)
        qcols2[0].metric("Total Volume", f"{quality['total_volume']:,d}")
        qcols2[1].metric("Total OI", f"{quality['total_open_interest']:,d}")
        qcols2[2].metric("Monotonicity Breaches", f"{quality['monotonicity_breaches']:,d}")
        qcols2[3].metric("Convexity Breaches", f"{quality['convexity_breaches']:,d}")

    st.markdown("")
    st.markdown("---")
    st.markdown("")
    st.markdown("#### Analytics Workbench")
    tabs = st.tabs(["Volatility", "Greeks", "P&L", "P&L Attribution", "MC vs BSM", "Risk Analysis"])

    with tabs[0]:
        st.subheader("Volatility")
        if clean_chain.empty:
            st.warning("No clean option data available after filtering. Try another expiry.")
        else:
            smile = clean_chain.copy()
            low_q, high_q = smile["iv_pct"].quantile([0.05, 0.95]).tolist()
            smile = smile[(smile["iv_pct"] >= low_q) & (smile["iv_pct"] <= high_q)]
            smile_fig = go.Figure()
            smile_fig.add_trace(go.Scatter(x=smile["strike"], y=smile["iv_pct"], mode="lines+markers", name="IV Smile"))
            smile_fig.add_vline(x=strike, line_dash="dash", line_color="orange", annotation_text="ATM")
            smile_fig.update_layout(
                title=f"{ticker} {option_type.title()} Volatility Smile",
                xaxis_title="Strike",
                yaxis_title="Implied Volatility (%)",
                template="plotly_white",
            )
            st.plotly_chart(smile_fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Greeks")
        greeks_table = {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [delta_val, gamma_val, vega_val, theta_val, rho_val],
        }
        st.dataframe(greeks_table, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("P&L")
        hedge_col1, hedge_col2 = st.columns(2)
        hedge_col1.metric("Option Delta", f"{delta_val:.4f}")
        hedge_col2.metric("Hedge Quantity", f"{hedge_qty:.4f}")
        pnl = simulate_delta_hedge_pnl(
            option_type=option_type,
            side=hedge_side,
            position=hedge_size,
            spot=spot,
            strike=strike,
            rate=rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
        )
        _render_hedging_charts(pnl, chart_view)

    with tabs[3]:
        st.subheader("P&L Attribution")
        dv = 0.01
        dr = 0.01
        ds = spot * 0.01
        d1 = delta_val * ds
        g1 = 0.5 * gamma_val * (ds**2)
        v1 = vega_val * dv
        r1 = rho_val * dr
        attribution = {
            "Component": ["Delta", "Gamma", "Vega", "Rho", "Approx Total"],
            "P&L": [d1, g1, v1, r1, d1 + g1 + v1 + r1],
        }
        st.dataframe(attribution, use_container_width=True, hide_index=True)
        st.caption("Attribution assumes +1% spot shock, +1 vol point, +100 bps rate shock.")

    with tabs[4]:
        st.subheader("MC vs BSM")
        diff = mc_price - bs_price
        mcols = st.columns(3)
        mcols[0].metric("BSM Price", f"{bs_price:.4f}")
        mcols[1].metric("Monte Carlo Price", f"{mc_price:.4f}")
        mcols[2].metric("MC - BSM", f"{diff:.4f}")

    with tabs[5]:
        st.subheader("Risk Analysis")
        st.metric("Spread Risk (median %)", f"{quality['median_spread_pct']:.2f}%")
        st.metric("Liquidity Depth (Volume + OI)", f"{quality['total_volume'] + quality['total_open_interest']}")
        risk_table = {
            "Check": ["Monotonicity", "Convexity", "IV-HV Spread"],
            "Value": [
                quality["monotonicity_breaches"],
                quality["convexity_breaches"],
                f"{iv_hv_spread:.2%}",
            ],
        }
        st.dataframe(risk_table, use_container_width=True, hide_index=True)

    st.markdown("")
    st.markdown("---")
    st.markdown("")

    with st.container(border=True):
        st.markdown('<div class="market-section-title">Cleaned Option Chain</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="market-section-subtitle">Filtered strikes used by analytics and risk charts</div>',
            unsafe_allow_html=True,
        )
        if clean_chain.empty:
            st.info("No rows after filters on this expiry. Displaying the first available raw rows for visibility.")
            raw_preview = raw_chain.sort_values("strike").head(40).copy()
            if "impliedVolatility" in raw_preview.columns:
                raw_preview["impliedVolatility"] = raw_preview["impliedVolatility"] * 100.0
            st.dataframe(raw_preview, use_container_width=True, hide_index=True)
        else:
            display_cols = [
                "strike",
                "bid",
                "ask",
                "mid_price",
                "spread_dollar",
                "spread_pct",
                "iv_pct",
                "volume",
                "openInterest",
            ]
            chain_view = clean_chain[[c for c in display_cols if c in clean_chain.columns]].copy()
            st.dataframe(
                chain_view,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                    "bid": st.column_config.NumberColumn("Bid", format="%.2f"),
                    "ask": st.column_config.NumberColumn("Ask", format="%.2f"),
                    "mid_price": st.column_config.NumberColumn("Mid", format="%.2f"),
                    "spread_dollar": st.column_config.NumberColumn("Spread $", format="%.3f"),
                    "spread_pct": st.column_config.NumberColumn("Spread %", format="%.2f"),
                    "iv_pct": st.column_config.NumberColumn("IV %", format="%.2f"),
                    "volume": st.column_config.NumberColumn("Volume", format="%d"),
                    "openInterest": st.column_config.NumberColumn("OI", format="%d"),
                },
            )


def render_market_news() -> None:
    """Display latest general market news with card layout."""
    st.markdown("## 📰 Market News Dashboard")
    st.caption("Latest financial headlines and market interpretation")

    refresh_col, _ = st.columns([1, 4])
    with refresh_col:
        if st.button("🔄 Refresh News", use_container_width=True):
            _get_news_cached.clear()
            _get_major_indices_cached.clear()
            st.success("News cache cleared.")

    st.caption("Cache TTL: 10 minutes")

    try:
        indices = _get_major_indices_cached()
        articles = _get_news_cached(limit=8)
    except RuntimeError as exc:
        st.error(f"Unable to fetch market news right now. Details: {exc}")
        return

    if indices:
        st.markdown("### Major Indices")
        idx_cols = st.columns(3)
        for i, quote in enumerate(indices):
            col = idx_cols[i % 3]
            col.metric(
                label=quote.name,
                value=f"{quote.last:,.2f}",
                delta=f"{quote.change:+.2f} ({quote.change_pct:+.2f}%)",
            )
        st.caption("Indicative Yahoo Finance snapshots (auto-refresh via cache).")
        st.markdown("---")

    if not articles:
        st.warning("⚠️ No recent general market news available at the moment.")
    else:
        for article in articles:
            safe_title = html.escape(article.title)
            safe_source = html.escape(article.source)
            safe_published = html.escape(article.published)
            safe_summary = html.escape(article.summary[:240] + ("..." if len(article.summary) > 240 else ""))
            st.markdown(
                f"""
                <div style="border:1px solid #e5e7eb; border-radius:14px; padding:16px; margin-bottom:14px; background-color:#f8fafc;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="background:#e0ecff; color:#1d4ed8; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600;">{safe_source}</span>
                        <span style="color:#6b7280; font-size:12px;">📅 {safe_published}</span>
                    </div>
                    <h4 style="margin:0 0 8px 0; color:#0f172a;">{safe_title}</h4>
                    <p style="margin:0 0 12px 0; color:#334155; font-size:14px;">{safe_summary or "Summary unavailable."}</p>
                    <a href="{article.link}" target="_blank" style="display:inline-block; text-decoration:none; background:#1d4ed8; color:white; padding:8px 12px; border-radius:8px; font-size:13px; font-weight:600;">Read article</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div style="border:1px solid #fde68a; border-radius:14px; padding:16px; margin-top:10px; background-color:#fffbeb;">
            <h4 style="margin:0 0 10px 0; color:#92400e;">📈 Market Interpretation</h4>
            <ul style="margin:0; padding-left:18px; color:#78350f;">
                <li>Macro uncertainty can increase implied volatility and lift option premiums.</li>
                <li>Rate-related headlines can shift rho sensitivity and discounting assumptions.</li>
                <li>Earnings and dense news flow can raise short-term option demand and repricing.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_barrier_products() -> None:
    """Barrier products pricer using local self-contained MC logic."""
    st.markdown("## Barrier Products")
    st.caption("Monte Carlo pricing for common barrier options and PDI-style structure.")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        with c2:
            rate = st.number_input("Risk-free rate", min_value=-0.05, value=0.04, step=0.0025, format="%.4f")
        with c3:
            dividend_yield = st.number_input("Dividend yield", min_value=0.0, value=0.0, step=0.0025, format="%.4f")

    with st.container(border=True):
        o1, o2 = st.columns(2)
        with o1:
            strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0)
        with o2:
            maturity = st.number_input("Maturity (years)", min_value=1e-4, value=1.0, step=0.1, format="%.4f")
        volatility = st.number_input("Volatility", min_value=1e-4, value=0.22, step=0.01, format="%.4f")

    with st.container(border=True):
        product = st.selectbox(
            "Product",
            options=[
                "Down-and-out call", "Up-and-out call", "Down-and-in call", "Up-and-in call",
                "Down-and-out put", "Up-and-out put", "Down-and-in put", "Up-and-in put",
                "PDI / Put Down-and-In",
            ],
            index=0,
        )
        if product == "PDI / Put Down-and-In":
            barrier_style = "down-and-in"
            option_type_for_model = "put"
        else:
            left, right = product.split(" ", 1)
            barrier_style = left.lower()
            option_type_for_model = right.lower()

        p1, p2 = st.columns(2)
        with p1:
            st.selectbox(
                "Option type",
                options=["call", "put"],
                index=0 if option_type_for_model == "call" else 1,
                disabled=True,
                key="bp_opt",
            )
        with p2:
            st.selectbox(
                "Barrier type",
                options=["down-and-out", "up-and-out", "down-and-in", "up-and-in"],
                index=["down-and-out", "up-and-out", "down-and-in", "up-and-in"].index(barrier_style),
                disabled=True,
                key="bp_bar",
            )

        if "down" in barrier_style:
            barrier_default = min(spot * 0.9, spot - 0.5)
            barrier = st.number_input("Barrier level", min_value=0.01, max_value=max(spot - 1e-4, 0.02), value=float(max(0.01, barrier_default)), step=0.5)
        else:
            barrier = st.number_input("Barrier level", min_value=spot + 1e-4, value=float(spot * 1.1), step=0.5)
        rebate = st.number_input("Rebate", min_value=0.0, value=0.0, step=0.1)

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        with m1:
            n_simulations = st.number_input("Simulations", min_value=1_000, value=40_000, step=1_000)
        with m2:
            n_steps = st.number_input("Time steps", min_value=2, value=252, step=10)
        with m3:
            use_bridge = st.checkbox("Brownian bridge correction", value=True)
        run_price = st.button("Price Barrier Product", type="primary", use_container_width=True)

    if not run_price:
        return

    try:
        result = price_barrier_option_mc(
            spot=float(spot),
            strike=float(strike),
            maturity=float(maturity),
            rate=float(rate),
            volatility=float(volatility),
            barrier=float(barrier),
            option_type=option_type_for_model,
            barrier_type=barrier_style,
            n_simulations=int(n_simulations),
            n_steps=int(n_steps),
            rebate=float(rebate),
            dividend_yield=float(dividend_yield),
            use_brownian_bridge=bool(use_bridge),
            seed=42,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    vanilla_ref = vanilla_black_scholes(
        spot=float(spot),
        strike=float(strike),
        maturity=float(maturity),
        rate=float(rate),
        volatility=float(volatility),
        option_type=option_type_for_model,
        dividend_yield=float(dividend_yield),
    )
    discount = result["price"] - vanilla_ref

    # Keep a sampled payoff distribution chart using identical dynamics.
    diag = price_barrier_option_mc(
        spot=float(spot),
        strike=float(strike),
        maturity=float(maturity),
        rate=float(rate),
        volatility=float(volatility),
        barrier=float(barrier),
        option_type=option_type_for_model,
        barrier_type=barrier_style,
        n_simulations=min(int(n_simulations), 20_000),
        n_steps=int(n_steps),
        rebate=float(rebate),
        dividend_yield=float(dividend_yield),
        use_brownian_bridge=bool(use_bridge),
        seed=7,
        return_payoffs=True,
    )

    with st.container(border=True):
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("Barrier Price", f"{result['price']:.4f}")
        r2.metric("Vanilla Ref", f"{vanilla_ref:.4f}")
        r3.metric("Barrier - Vanilla", f"{discount:+.4f}")
        r4.metric("Std Error", f"{result['std_error']:.6f}")
        r5.metric("95% CI", f"[{result['ci_95_low']:.4f}, {result['ci_95_high']:.4f}]")
        if "out" in barrier_style:
            r6.metric("Knock-out Prob", f"{result['knock_out_probability']*100:.2f}%")
        else:
            r6.metric("Knock-in Prob", f"{result['knock_in_probability']*100:.2f}%")

        st.caption(product_payoff_explanation(option_type_for_model, barrier_style, float(rebate)))

        summary = pd.DataFrame(
            [
                {"Metric": "Product", "Value": product},
                {"Metric": "Spot / Strike / Barrier", "Value": f"{spot:.2f} / {strike:.2f} / {barrier:.2f}"},
                {"Metric": "Maturity / Vol", "Value": f"{maturity:.4f} / {volatility*100:.2f}%"},
                {"Metric": "Rate / Dividend", "Value": f"{rate*100:.2f}% / {dividend_yield*100:.2f}%"},
                {"Metric": "MC sims x steps", "Value": f"{int(n_simulations)} x {int(n_steps)}"},
                {"Metric": "Hit Probability", "Value": f"{result['hit_probability']*100:.2f}%"},
                {"Metric": "Price", "Value": f"{result['price']:.6f}"},
                {"Metric": "Std Error", "Value": f"{result['std_error']:.6f}"},
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        discounted_payoffs = np.asarray(diag.get("discounted_payoffs", np.array([])), dtype=float)
        if discounted_payoffs.size > 5000:
            rng = np.random.default_rng(123)
            idx = rng.choice(discounted_payoffs.size, size=5000, replace=False)
            discounted_payoffs = discounted_payoffs[idx]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=discounted_payoffs, nbinsx=60, marker_color="#1d4ed8", opacity=0.85))
        fig_hist.add_vline(x=result["price"], line_dash="dash", line_color="#f59e0b", annotation_text="Barrier price")
        fig_hist.add_vline(x=vanilla_ref, line_dash="dot", line_color="#94a3b8", annotation_text="Vanilla ref")
        fig_hist.update_layout(
            title="Payoff distribution (discounted, MC approximation)",
            xaxis_title="Discounted payoff",
            yaxis_title="Frequency",
            template="plotly_white",
            height=340,
        )
        st.plotly_chart(fig_hist, use_container_width=True)


def render_structured_products() -> None:
    """Structuring budget lab: ZC funding, short PDI slice, digital leg, then full-note reconciliation."""
    st.markdown("## Structured Products Lab")
    st.caption("Construis et price un produit structure etape par etape : funding, downside, coupon.")

    if "struct_lab_coupon_pct" not in st.session_state:
        st.session_state.struct_lab_coupon_pct = 8.5
    if "struct_last_calibrated_coupon_pct" not in st.session_state:
        st.session_state.struct_last_calibrated_coupon_pct = None

    # --- Section 1: Zero-coupon / funding ---
    with st.container(border=True):
        st.subheader("1) Zero-coupon bond / funding")
        f1, f2, f3 = st.columns(3)
        with f1:
            notional = st.number_input("Notional (currency)", min_value=1e3, value=1_000_000.0, step=50_000.0)
            maturity = st.number_input("Maturity (years)", min_value=0.1, value=3.0, step=0.25)
        with f2:
            rate = st.number_input("Risk-free rate r", min_value=-0.02, value=0.04, step=0.0025, format="%.4f")
            issuer_spread = st.number_input(
                "Funding / issuer spread s (added to discount)",
                min_value=0.0,
                value=0.0025,
                step=0.0005,
                format="%.4f",
            )
        with f3:
            # Par issue: investor pays 100% of notional; target PV for structuring = N (not a discount/premium issue).
            target_issue_pct = 100.0
            st.caption("Prix d'émission cible")
            st.markdown("**100 % du notionnel** (émission à par)")
        issue_price_pct = float(target_issue_pct)

        fund = funding_leg(float(notional), float(maturity), float(rate), float(issuer_spread), float(target_issue_pct))
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("ZC price (% notional)", f"{fund['zc_pct_of_notional']:.2f}%")
        o2.metric("ZC PV (currency)", f"{fund['zc_pv']:,.0f}")
        o3.metric("Option budget (target − ZC)", f"{fund['option_budget_from_zc']:,.0f}")
        o4.metric("Option budget (% N)", f"{fund['option_budget_pct_of_notional']:.2f}%")
        st.info(
            "**Higher r or wider funding spread ⇒ lower df(T) ⇒ lower ZC PV ⇒ larger option budget** for a **par** "
            "issue (target PV = 100 % of notional), i.e. more PV room for coupons / upside after backing out the bullet."
        )

    # --- Section 2: Short PDI / downside slice (single underlying) ---
    with st.container(border=True):
        st.subheader("2) Short PDI / downside risk (single underlying)")
        p1, p2, p3 = st.columns(3)
        with p1:
            initial_level = st.number_input("Initial level", min_value=1e-6, value=100.0)
        with p2:
            vol = st.number_input("ATM volatility", min_value=1e-4, value=0.25, step=0.01)
            skew_steep = st.slider("Skew steepness (higher vol when spot is lower)", 0.05, 1.2, 0.45, 0.05)
            divy = st.number_input("Dividend yield q", min_value=0.0, value=0.02, step=0.0025, format="%.4f")
        with p3:
            ki_barrier = st.slider("Knock-in barrier (% of ref)", 30.0, 100.0, 70.0, 0.5)
            pdi_strike_pct = st.slider("PDI strike (% of initial)", 50.0, 120.0, 100.0, 0.5)
            ki_style = st.radio("KI observation style", ["discrete", "continuous"], index=1, horizontal=True)
        st.caption(
            "Classic short Put Down-and-In financing leg: if KI is never touched payoff is 0; if touched, "
            "payoff is max(PDI strike - S_T, 0)."
        )
        # No standalone spot input in this simplified PDI setup.
        spot = float(initial_level)

    # --- Section 3: Digital coupon + full note harness ---
    with st.container(border=True):
        st.subheader("3) Digital coupon / autocall leg (full engine)")
        d1, d2 = st.columns(2)
        with d1:
            product_label = st.selectbox(
                "Product template",
                options=[
                    "Athena autocall",
                    "Phoenix autocall — memory coupon",
                    "Phoenix autocall — non-memory coupon",
                    "Worst-of Phoenix — memory coupon",
                    "Worst-of Phoenix — non-memory coupon",
                ],
            )
            worst_of = product_label.startswith("Worst-of")
            if "Athena" in product_label:
                family = "athena_autocall"
            elif "non-memory" in product_label.lower():
                family = "phoenix_autocall_no_memory"
            elif "memory" in product_label.lower():
                family = "phoenix_autocall_memory"
            else:
                family = "phoenix_autocall_no_memory"
        is_athena = family == "athena_autocall"
        with d2:
            if is_athena:
                athena_barrier = st.slider(
                    "Autocall / coupon barrier — Athena (% of ref)",
                    50.0,
                    150.0,
                    100.0,
                    0.5,
                    help="Athena: no Phoenix coupons; the coupon is paid only on early redemption, "
                    "with the same level as the autocall barrier.",
                )
                coupon_barrier = float(athena_barrier)
                autocall_barrier = float(athena_barrier)
                st.caption("Single barrier: interim digital ≠ Phoenix; coupon attaches to autocall only.")
            else:
                coupon_barrier = st.slider("Coupon / digital barrier (% of ref)", 50.0, 120.0, 95.0, 0.5)
                autocall_barrier = st.slider("Autocall barrier (% of ref)", 80.0, 150.0, 110.0, 0.5)
        dc1, dc2 = st.columns(2)
        with dc1:
            coupon_annual = (
                st.slider(
                    "Headline annual coupon (% of notional)",
                    0.1,
                    40.0,
                    key="struct_lab_coupon_pct",
                )
                / 100.0
            )
            frequency = st.selectbox("Observation frequency", ["monthly", "quarterly", "semi_annual", "annual"])
        with dc2:
            # Fixed MC harness for structured desk-like setup: 50k paths, daily time steps.
            n_sims = 50_000
            n_steps = max(2, int(round(252 * float(maturity))))
            # Convention fixed for all current templates.
            coupon_convention = "full_headline_per_date"
        n_assets = st.number_input("Worst-of constituents", min_value=2, max_value=6, value=3, disabled=not worst_of)
        correlation = st.slider("Pairwise correlation ρ", 0.05, 0.95, 0.55, disabled=not worst_of)

    bump_cfg = dict(
        notional=float(notional),
        spot=float(spot),
        initial_level=float(initial_level),
        # Structured lab no longer exposes a separate strike input: anchor strike to initial level.
        strike=float(initial_level),
        maturity_years=float(maturity),
        risk_free_rate=float(rate),
        dividend_yield=float(divy),
        issuer_spread=float(issuer_spread),
        atm_volatility=float(vol),
        skew_steepness=float(skew_steep),
        correlation=float(correlation if worst_of else 0.35),
        autocall_barrier_ratio=float(autocall_barrier) / 100.0,
        coupon_barrier_ratio=float(coupon_barrier) / 100.0,
        knock_in_barrier_ratio=float(ki_barrier) / 100.0,
        capital_recovery_barrier_ratio=float(pdi_strike_pct) / 100.0,
        coupon_level_annual_pct=float(coupon_annual),
        coupon_convention=coupon_convention,
        observation_frequency=frequency,
        family=family,
        use_worst_of=bool(worst_of),
        n_assets=int(n_assets if worst_of else 1),
        n_simulations=int(n_sims),
        n_time_steps=int(n_steps),
        issue_price_pct=float(issue_price_pct),
        redemption_floor_ratio=0.0,
    )

    struct_seed = 771_004
    struct_inp = StructuredPricingInput(**bump_cfg)

    if st.button("Calibrate coupon to target", type="secondary"):
        from core.structured_products import calibrate_coupon_for_target

        tgt_pv = struct_inp.notional * float(target_issue_pct) / 100.0
        cal_c, meta = calibrate_coupon_for_target(struct_inp, tgt_pv, seed=struct_seed)
        st.session_state.struct_lab_coupon_pct = float(min(40.0, max(0.1, cal_c * 100.0)))
        st.session_state.struct_last_calibrated_coupon_pct = float(cal_c * 100.0)
        if meta.get("status") == "above_bracket":
            st.warning("Target required coupon above search bracket.")
        st.rerun()

    with st.spinner("Monte Carlo (PDI slice + digital leg + full note) ..."):
        pdi = mcp_pdi_slice(
            notional=float(notional),
            initial_level=float(initial_level),
            maturity_years=float(maturity),
            risk_free_rate=float(rate),
            dividend_yield=float(divy),
            issuer_spread=float(issuer_spread),
            atm_vol=float(vol),
            skew_steepness=float(skew_steep),
            ki_barrier_ratio=float(ki_barrier) / 100.0,
            pdi_strike_ratio=float(pdi_strike_pct) / 100.0,
            n_paths=int(n_sims),
            n_steps=int(n_steps),
            frequency=frequency,
            ki_style=ki_style,
            seed=struct_seed + 3,
        )
        digital = digital_leg_from_note(struct_inp, seed=struct_seed + 5)
        valuation = price_structured_product(struct_inp, seed=struct_seed + 7)
        wf = structuring_waterfall(fund, pdi, digital, float(valuation["fair_mean"]), float(notional))

    # --- Section 4: Structuring summary ---
    with st.container(border=True):
        st.subheader("4) Structuring summary")
        gap = wf["gap_to_target_pct_pts"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Protection du capital (ZC)", f"{fund['zc_pv']:,.0f}")
        s2.metric("Budget options disponible", f"{wf['option_budget_initial']:,.0f}")
        s3.metric("Gap to target", f"{gap:+.2f} pts")

        if abs(gap) < 0.5:
            st.success(f"Gap status: {gap:+.2f} pts (vert)")
        elif abs(gap) < 2.0:
            st.warning(f"Gap status: {gap:+.2f} pts (orange)")
        else:
            st.error(f"Gap status: {gap:+.2f} pts (rouge)")

        with st.expander("Détail du waterfall", expanded=False):
            opt_spent = float(digital["digital_pv"]) - float(pdi["premium_short_pdi"])
            opt_budget = float(wf["option_budget_initial"])
            opt_left = opt_budget - opt_spent

            d1, d2, d3 = st.columns(3)
            d1.metric("PDI premium (short, +)", f"{pdi['premium_short_pdi']:,.0f}")
            d2.metric("Digital / coupon leg PV", f"{digital['digital_pv']:,.0f}")
            d3.metric("Net option package (dig − PDI)", f"{wf['net_option_package_pv']:,.0f}")

            d4, d5, d6 = st.columns(3)
            d4.metric("Budget options utilisé", f"{opt_spent:,.0f}")
            d5.metric("Budget options restant", f"{opt_left:,.0f}")
            d6.metric("Full note fair PV", f"{wf['fair_mc']:,.0f}")

            t1, _, _ = st.columns(3)
            t1.metric("Target issue PV", f"{fund['target_pv']:,.0f}")

            st.markdown(
                "**Budget options (même échelle devise / notionnel) :** "
                f"disponible **{opt_budget:,.0f}** → utilisé **{opt_spent:,.0f}** (digital PV − prime PDI short) → "
                f"restant **{opt_left:,.0f}**. "
                "Le **Gap to target** ci-dessus reste celui du Monte Carlo sur la note complète."
            )

    with st.expander("Section notes (educational)", expanded=False):
        st.markdown(
            """
**§1 — ZC:** Bullet principal at T discounted at r+s. This is the backbone PV before any options.

**§2 — Short PDI downside option:** Isolated single-name MC with equity skew (vol up when spot is down). Selling this
downside generates PV (`premium_short_pdi`) that can fund digitals, but creates crash risk if KI and spot is weak.

**§3 — Digital leg:** Uses the full structured note Monte Carlo so Phoenix memory / Athena autocall / worst-of
remain consistent; the **digital PV** line is the engine’s discounted coupon stream (including autocall coupon leg).

**§4 — Summary:** The additive budget identity is approximate because the same paths enter coupons and principal;
the **full fair value** is the authoritative PV from the combined engine.
            """
        )

    # PDI diagnostics block
    with st.container(border=True):
        st.subheader("PDI slice diagnostics")
        q1, q3 = st.columns(2)
        q1.metric("PDI premium (received)", f"{pdi['premium_short_pdi']:,.0f}")
        loss_if_ki = pdi["expected_loss_abs_given_ki"]
        premium_rcv = pdi["premium_short_pdi"]
        if math.isnan(loss_if_ki) or premium_rcv <= 0:
            ratio_txt = "N/A"
        else:
            ratio_txt = f"soit {100.0 * loss_if_ki / premium_rcv:,.1f}% de la prime recue"
        q3.metric(
            "Perte moyenne si KI touche",
            "N/A" if math.isnan(loss_if_ki) else f"{loss_if_ki:,.0f}",
            ratio_txt,
        )

    with st.container(border=True):
        st.subheader("Digital leg diagnostics")
        d1, d2 = st.columns(2)
        d1.metric("Digital PV (% notional)", f"{digital['digital_pv_pct_notional']:.2f}%")
        d2.metric("Autocall probability", f"{digital['prob_autocall']*100:.2f}%")
        if family.startswith("phoenix"):
            d3 = st.columns(1)[0]
            prob_ph = digital.get("prob_all_coupons")
            d3.metric(
                "Coupon completion (Phoenix)",
                "N/A" if (prob_ph is None or math.isnan(prob_ph)) else f"{prob_ph*100:.2f}%",
            )
        st.caption(
            "Digital coupons are financed by the ZC budget and by the premium from selling downside; raising "
            "coupon or lowering barriers consumes more PV."
        )

    with st.container(border=True):
        st.subheader("Sensitivities & Greeks (full note)")
        from core.structured_products import monte_carlo_greeks, sensitivity_table

        sensi = sensitivity_table(struct_inp)
        greeks_quick = monte_carlo_greeks(
            StructuredPricingInput(**{**bump_cfg, "n_simulations": min(6_000, int(n_sims))}),
            seed=struct_seed,
        )
        simple_rows = [
            ("Delta (sensibilite au spot)", sensi.get("fair_value_pv_per_$spot", float("nan"))),
            ("Vega (par pt de vol)", sensi.get("fair_value_pv_per_1vol_pt", float("nan"))),
            ("Rho (par bp de taux)", sensi.get("fair_value_pv_per_bp_rate", float("nan"))),
            ("Div sensitivity (par bp de dividende)", sensi.get("fair_value_pv_per_bp_div", float("nan"))),
        ]
        st.dataframe(
            pd.DataFrame(simple_rows, columns=["Sensibilite", "Valeur"]),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Greeks avances", expanded=False):
            advanced_keys = [
                "fair_value_pv_per_ki_barrier",
                "fair_value_pv_per_coupon_barrier",
                "fair_value_pv_per_skew_unit",
                "fair_value_pv_per_autocall_barrier",
                "fair_value_pv_per_cap_barrier",
            ]
            adv = {k: sensi.get(k, float("nan")) for k in advanced_keys}
            adv.update(greeks_quick)
            st.dataframe(pd.DataFrame([adv]), use_container_width=True)

    st.markdown("---")

def main() -> None:
    st.set_page_config(page_title="Structured Options Pricer", layout="wide")
    st.title("Structured Options Pricer")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            border-right: 1px solid #dbe3ee;
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stRadio label {
            color: #334155;
            font-weight: 600;
            font-size: 0.9rem;
        }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            background-color: #ffffff;
        }
        .sidebar-section-chip {
            border: 1px solid #cbd5e1;
            border-left-width: 4px;
            border-radius: 10px;
            padding: 8px 10px;
            margin-bottom: 12px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    section_themes = {
        "Market News": {"bg": "#0f172a", "fg": "#f8fafc", "accent": "#f59e0b"},
        "Simple Pricer": {"bg": "#ffffff", "fg": "#0f172a", "accent": "#2563eb"},
        "Market Vanilla Options": {"bg": "#ffffff", "fg": "#0f172a", "accent": "#0f766e"},
        "Barrier Products": {"bg": "#ffffff", "fg": "#0f172a", "accent": "#9333ea"},
        "Structured Products Lab": {"bg": "#ffffff", "fg": "#0f172a", "accent": "#b45309"},
    }

    app_section = st.sidebar.selectbox(
        "App Section",
        options=["Market News", "Simple Pricer", "Market Vanilla Options", "Barrier Products", "Structured Products Lab"],
        index=0,
    )
    theme = section_themes[app_section]
    st.sidebar.markdown(
        (
            f'<div class="sidebar-section-chip" style="background:{theme["bg"]};'
            f'color:{theme["fg"]}; border-left-color:{theme["accent"]}; border-color:{theme["accent"]}33;">'
            f"Active section: {app_section}</div>"
        ),
        unsafe_allow_html=True,
    )

    if app_section == "Market News":
        render_market_news()
    elif app_section == "Simple Pricer":
        render_simple_pricer()
    elif app_section == "Market Vanilla Options":
        render_market_vanilla_options()
    elif app_section == "Barrier Products":
        render_barrier_products()
    elif app_section == "Structured Products Lab":
        render_structured_products()


if __name__ == "__main__":
    main()
