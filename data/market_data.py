"""Yahoo Finance market data utilities for vanilla options."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf


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
        atm_iv = float(atm_row["impliedVolatility"])

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
