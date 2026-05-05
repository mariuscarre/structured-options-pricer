"""General market news utilities backed by financial RSS feeds."""

from __future__ import annotations

from dataclasses import dataclass

import feedparser
import requests


@dataclass
class NewsArticle:
    """Normalized market news article."""

    title: str
    source: str
    link: str
    published: str
    summary: str


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
