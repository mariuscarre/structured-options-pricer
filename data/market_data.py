"""Yahoo Finance market data utilities for vanilla options."""

from __future__ import annotations

import math
from datetime import date, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from core.black_scholes import call_price, put_price


@dataclass
class MarketOptionData:
    """Container for one ticker's option market snapshot."""

    ticker: str
    spot_price: float
    expirations: list[str]
    expiration: str
    option_type: str
    option_chain: pd.DataFrame
    atm_strike: float
    atm_iv: float


def _extract_spot_price(ticker_obj: yf.Ticker) -> float:
    """Try fast and fallback fields for current spot price."""
    info = ticker_obj.fast_info
    spot = info.get("last_price") or info.get("regular_market_price")
    if spot is None:
        hist = ticker_obj.history(period="1d")
        if hist.empty:
            raise RuntimeError("Unable to fetch current stock price.")
        spot = float(hist["Close"].iloc[-1])
    return float(spot)


def _prepare_chain(option_chain: pd.DataFrame) -> pd.DataFrame:
    """Keep only useful interview-friendly columns."""
    preferred_columns = [
        "contractSymbol",
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
        "inTheMoney",
    ]
    available_columns = [col for col in preferred_columns if col in option_chain.columns]
    chain = option_chain[available_columns].copy()
    chain = chain.sort_values("strike").reset_index(drop=True)
    return chain


def _closest_strike_row(option_chain: pd.DataFrame, spot_price: float) -> pd.Series:
    """Return option row with strike nearest to spot."""
    idx = (option_chain["strike"] - spot_price).abs().idxmin()
    return option_chain.loc[idx]


def _robust_atm_iv(option_chain: pd.DataFrame, spot_price: float) -> float:
    """Compute a robust ATM IV around spot, ignoring clearly bad quotes."""
    chain = option_chain.copy()

    # Keep rows that carry at least some market signal and sane IV levels.
    has_price_signal = (chain["bid"] > 0) | (chain["ask"] > 0) | (chain["lastPrice"] > 0)
    sane_iv = chain["impliedVolatility"].notna() & (chain["impliedVolatility"] >= 0.01) & (chain["impliedVolatility"] <= 3.0)
    liquid = chain[has_price_signal & sane_iv].copy()

    # Focus near ATM first; if too sparse, widen progressively.
    for band in (0.10, 0.20, 0.35):
        near = liquid[(liquid["strike"] >= (1.0 - band) * spot_price) & (liquid["strike"] <= (1.0 + band) * spot_price)]
        if len(near) >= 3:
            # Median is robust to stale/outlier quotes.
            return float(near["impliedVolatility"].median())

    if not liquid.empty:
        closest = _closest_strike_row(liquid, spot_price)
        return float(closest["impliedVolatility"])

    # Last fallback: closest strike from raw chain, even if noisy.
    fallback = _closest_strike_row(chain, spot_price)
    return float(fallback["impliedVolatility"])


def fetch_market_option_data(ticker: str, expiration: str, option_type: str) -> MarketOptionData:
    """Fetch spot, expirations, option chain, ATM strike and ATM IV from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker)
        spot = _extract_spot_price(ticker_obj)
        expirations = list(ticker_obj.options)
        if not expirations:
            raise RuntimeError(f"No option expirations found for {ticker}.")

        selected_expiration = expiration if expiration in expirations else expirations[0]
        chain_obj = ticker_obj.option_chain(selected_expiration)
        raw_chain = chain_obj.calls if option_type == "call" else chain_obj.puts
        if raw_chain.empty:
            raise RuntimeError(f"No {option_type} options found for {ticker} at {selected_expiration}.")

        chain = _prepare_chain(raw_chain)
        atm_row = _closest_strike_row(chain, spot)
        atm_strike = float(atm_row["strike"])
        atm_iv = _robust_atm_iv(chain, spot)

        return MarketOptionData(
            ticker=ticker,
            spot_price=spot,
            expirations=expirations,
            expiration=selected_expiration,
            option_type=option_type,
            option_chain=chain,
            atm_strike=atm_strike,
            atm_iv=atm_iv,
        )
    except Exception as exc:
        raise RuntimeError(f"Market data fetch failed for {ticker}: {exc}") from exc


def fetch_historical_volatility(ticker: str, window: int = 20) -> float:
    """Estimate annualized historical volatility from daily closes."""
    try:
        ticker_obj = yf.Ticker(ticker)
        history = ticker_obj.history(period="3mo")
        if history.empty or "Close" not in history:
            raise RuntimeError(f"No historical data available for {ticker}.")

        returns = history["Close"].pct_change().dropna()
        if len(returns) < max(window, 5):
            raise RuntimeError(f"Not enough historical observations for {ticker}.")

        hv = returns.tail(window).std() * (252.0**0.5)
        return float(hv)
    except Exception as exc:
        raise RuntimeError(f"Historical volatility fetch failed for {ticker}: {exc}") from exc


def _synthetic_expirations() -> list[str]:
    """Build a short synthetic expiration strip (weekly then monthly-like)."""
    today = date.today()
    day_offsets = [7, 14, 21, 30, 45, 60, 90, 120, 180]
    return sorted({(today + timedelta(days=d)).strftime("%Y-%m-%d") for d in day_offsets})


def _enforce_static_noarb(strikes: np.ndarray, mids: np.ndarray, option_type: str, rate: float, t_years: float) -> np.ndarray:
    """Project synthetic prices to monotone+convex shape under static no-arbitrage bounds."""
    k = np.asarray(strikes, dtype=float)
    p = np.maximum(np.asarray(mids, dtype=float), 0.0)
    if len(k) < 3:
        return p

    dk = np.diff(k)
    valid = dk > 1e-12
    if not np.all(valid):
        return p

    df = math.exp(-max(rate, 0.0) * max(t_years, 0.0))
    slopes = np.diff(p) / dk
    if option_type == "put":
        # dP/dK in [0, df], non-decreasing in K for convexity.
        slopes = np.clip(slopes, 0.0, df)
    else:
        # dC/dK in [-df, 0], non-decreasing in K for convexity.
        slopes = np.clip(slopes, -df, 0.0)

    slopes = np.maximum.accumulate(slopes)
    out = np.zeros_like(p)
    out[0] = max(p[0], 0.0)
    for i in range(len(slopes)):
        out[i + 1] = out[i] + slopes[i] * dk[i]
    return np.maximum(out, 0.0)


def fetch_synthetic_option_data(ticker: str, expiration: str, option_type: str, rate: float = 0.03) -> MarketOptionData:
    """Generate synthetic, smooth option market data for stable demos/debug."""
    spot_map = {
        "AAPL": 190.0,
        "MSFT": 415.0,
        "TSLA": 185.0,
        "NVDA": 980.0,
        "AMZN": 182.0,
        "GOOGL": 165.0,
    }
    spot = float(spot_map.get(ticker.upper(), 200.0))
    expirations = _synthetic_expirations()
    selected_expiration = expiration if expiration in expirations else expirations[0]

    today = date.today()
    expiry = date.fromisoformat(selected_expiration)
    t_years = max((expiry - today).days / 365.0, 1.0 / 365.0)

    strikes = np.linspace(0.7 * spot, 1.3 * spot, 61)
    moneyness = strikes / max(spot, 1e-12)
    log_m = np.log(np.maximum(moneyness, 1e-12))

    # Equity-like vol surface (stylized facts):
    # - negative skew (OTM puts richer IV than OTM calls)
    # - asymmetric wings (put wing steeper than call wing)
    # - convexity (surface is not a straight line)
    base_iv = 0.20 + 0.05 * min(t_years, 1.0)
    linear_skew = -0.18 * log_m
    convex = 0.22 * (log_m**2)
    put_wing = 0.28 * np.maximum(-log_m, 0.0) ** 1.45
    call_wing = 0.08 * np.maximum(log_m, 0.0) ** 1.25
    iv = base_iv + linear_skew + convex + put_wing + call_wing
    iv = np.clip(iv, 0.08, 0.85)

    if option_type == "call":
        mids = np.array([call_price(spot, float(k), rate, float(sig), t_years) for k, sig in zip(strikes, iv)])
    else:
        mids = np.array([put_price(spot, float(k), rate, float(sig), t_years) for k, sig in zip(strikes, iv)])
    mids = _enforce_static_noarb(strikes, mids, option_type=option_type, rate=rate, t_years=t_years)
    mids = np.maximum(mids, 1e-4)

    # Deterministic microstructure noise by key for reproducibility.
    rng_seed = abs(hash((ticker.upper(), selected_expiration, option_type))) % (2**32)
    rng = np.random.default_rng(rng_seed)
    spread_abs = np.maximum(2e-4, 0.01 * np.sqrt(np.maximum(mids, 1e-8)))
    spread_abs *= rng.uniform(0.9, 1.15, size=len(spread_abs))
    spread_abs = np.minimum(spread_abs, 0.08 * mids)

    bids = np.maximum(mids - 0.5 * spread_abs, 1e-6)
    asks = np.maximum(mids + 0.5 * spread_abs, bids + 1e-6)
    last = np.maximum(mids * rng.uniform(0.985, 1.015, size=len(mids)), 0.0)
    vol = rng.integers(40, 3000, size=len(strikes))
    oi = rng.integers(100, 12000, size=len(strikes))
    itm = strikes < spot if option_type == "call" else strikes > spot

    chain = pd.DataFrame(
        {
            "contractSymbol": [f"{ticker.upper()}_{selected_expiration.replace('-', '')}_{option_type.upper()}_{int(round(k*1000))}" for k in strikes],
            "strike": strikes,
            "lastPrice": last,
            "bid": bids,
            "ask": asks,
            "volume": vol,
            "openInterest": oi,
            "impliedVolatility": iv,
            "inTheMoney": itm,
        }
    ).sort_values("strike").reset_index(drop=True)

    atm_row = _closest_strike_row(chain, spot)
    return MarketOptionData(
        ticker=ticker.upper(),
        spot_price=spot,
        expirations=expirations,
        expiration=selected_expiration,
        option_type=option_type,
        option_chain=chain,
        atm_strike=float(atm_row["strike"]),
        atm_iv=float(_robust_atm_iv(chain, spot)),
    )
