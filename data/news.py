"""General market news utilities backed by financial RSS feeds."""

from __future__ import annotations

from dataclasses import dataclass

import feedparser
import requests
try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional at runtime on some deployments
    yf = None


@dataclass
class NewsArticle:
    """Normalized market news article."""

    title: str
    source: str
    link: str
    published: str
    summary: str


@dataclass
class IndexQuote:
    """Snapshot for a major equity index."""

    name: str
    symbol: str
    last: float
    change: float
    change_pct: float


def _parse_feed_articles(feed_text: str, source: str, limit: int) -> list[NewsArticle]:
    """Parse RSS feed text into normalized article entries."""
    parsed_feed = feedparser.parse(feed_text)
    articles: list[NewsArticle] = []
    for entry in parsed_feed.entries:
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        published = entry.get("published", "Date unavailable").strip()
        summary = entry.get("summary", "").strip()
        if title and link:
            articles.append(NewsArticle(title=title, source=source, link=link, published=published, summary=summary))
        if len(articles) >= limit:
            break
    return articles


def _fetch_rss_articles(url: str, source: str) -> list[NewsArticle]:
    """Fetch one RSS feed and return parsed articles."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    # Parse generously here, then trim globally after merge.
    return _parse_feed_articles(response.text, source=source, limit=20)


def fetch_general_market_news(limit: int = 8) -> list[NewsArticle]:
    """Fetch latest general market headlines across reliable RSS feeds."""
    feeds = [
        ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC Markets RSS"),
        ("http://feeds.marketwatch.com/marketwatch/topstories/", "MarketWatch Top Stories RSS"),
        ("https://finance.yahoo.com/news/rssindex", "Yahoo Finance General RSS"),
    ]
    merged: list[NewsArticle] = []

    for feed_url, source in feeds:
        try:
            merged.extend(_fetch_rss_articles(feed_url, source=source))
        except Exception:
            continue

    # Deduplicate by link while preserving order.
    unique_articles: list[NewsArticle] = []
    seen_links: set[str] = set()
    for article in merged:
        if article.link in seen_links:
            continue
        seen_links.add(article.link)
        unique_articles.append(article)
        if len(unique_articles) >= limit:
            break

    return unique_articles


def _safe_float(value: object) -> float | None:
    """Best-effort float conversion."""
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:  # NaN guard
        return None
    return out


def _fetch_index_quote(symbol: str, name: str) -> IndexQuote | None:
    """Fetch one index quote from Yahoo with resilient fallbacks."""
    if yf is None:
        return None
    try:
        ticker = yf.Ticker(symbol)

        last = _safe_float(ticker.fast_info.get("lastPrice"))
        prev_close = _safe_float(ticker.fast_info.get("previousClose"))

        if last is None or prev_close is None or prev_close <= 0:
            hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
            if hist is None or hist.empty:
                return None
            close_series = hist["Close"].dropna()
            if len(close_series) == 0:
                return None
            last = float(close_series.iloc[-1])
            prev_close = float(close_series.iloc[-2]) if len(close_series) >= 2 else last
            if prev_close <= 0:
                return None

        change = last - prev_close
        change_pct = (change / prev_close) * 100.0
        return IndexQuote(name=name, symbol=symbol, last=last, change=change, change_pct=change_pct)
    except Exception:
        return None


def fetch_major_indices() -> list[IndexQuote]:
    """Fetch key global equity index snapshots."""
    index_map = [
        ("S&P 500", "^GSPC"),
        ("Nasdaq 100", "^NDX"),
        ("Dow Jones", "^DJI"),
        ("Euro Stoxx 50", "^STOXX50E"),
        ("CAC 40", "^FCHI"),
        ("DAX", "^GDAXI"),
        ("FTSE 100", "^FTSE"),
        ("Nikkei 225", "^N225"),
        ("Hang Seng", "^HSI"),
    ]
    quotes: list[IndexQuote] = []
    for name, symbol in index_map:
        quote = _fetch_index_quote(symbol=symbol, name=name)
        if quote is not None:
            quotes.append(quote)
    return quotes
