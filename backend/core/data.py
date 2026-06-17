"""Deterministic market-data access via yfinance.

Fetches adjusted-close price history for a set of tickers (and, optionally, a
benchmark) and returns a tidy price matrix plus the list of tickers that had
no data, so callers can degrade gracefully. No metrics are computed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import yfinance as yf

from ..models.schemas import Portfolio

__all__ = [
    "DEFAULT_PERIOD",
    "DEFAULT_INTERVAL",
    "MarketDataError",
    "PriceHistory",
    "fetch_price_history",
    "get_portfolio_prices",
    "fetch_benchmark_history",
    "clear_cache",
]

DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"

# Simple in-process cache so repeated questions in one session don't re-download.
_CACHE: Dict[Tuple[Tuple[str, ...], str, str], pd.DataFrame] = {}


class MarketDataError(RuntimeError):
    """Raised when no price data could be retrieved for any requested ticker."""


@dataclass
class PriceHistory:
    """Adjusted-close prices for the tickers that returned data."""

    prices: pd.DataFrame                      # index = dates, columns = tickers
    missing: List[str] = field(default_factory=list)
    period: str = DEFAULT_PERIOD
    interval: str = DEFAULT_INTERVAL

    @property
    def available(self) -> List[str]:
        return list(self.prices.columns)

    def __bool__(self) -> bool:
        return not self.prices.empty


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def fetch_price_history(
    tickers: Sequence[str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    *,
    use_cache: bool = True,
) -> PriceHistory:
    """Download adjusted-close history for ``tickers``.

    Missing/invalid symbols are reported in ``PriceHistory.missing`` rather than
    raising — only a *total* failure (no data for any ticker) raises.
    """
    cleaned = _unique_upper(tickers)
    if not cleaned:
        raise MarketDataError("No tickers provided.")

    key = (tuple(cleaned), period, interval)
    if use_cache and key in _CACHE:
        close = _CACHE[key]
    else:
        close = _download_adjusted_close(cleaned, period, interval)
        if use_cache:
            _CACHE[key] = close

    present = [t for t in cleaned if t in close.columns and close[t].notna().any()]
    missing = [t for t in cleaned if t not in present]

    prices = close.reindex(columns=present).dropna(how="all")
    if prices.empty:
        raise MarketDataError(
            f"No price data returned for any ticker: {cleaned}. "
            "Check the symbols (NSE tickers need a '.NS' suffix) or your connection."
        )
    return PriceHistory(prices=prices, missing=missing, period=period, interval=interval)


def get_portfolio_prices(
    portfolio: Portfolio,
    period: str = DEFAULT_PERIOD,
    *,
    use_cache: bool = True,
) -> PriceHistory:
    """Convenience: fetch adjusted-close history for a Portfolio's tickers."""
    return fetch_price_history(portfolio.tickers, period=period, use_cache=use_cache)


def fetch_benchmark_history(
    benchmark: str = "^NSEI",
    period: str = DEFAULT_PERIOD,
    *,
    use_cache: bool = True,
) -> PriceHistory:
    """Convenience: fetch a single benchmark series (default Nifty 50, ^NSEI)."""
    return fetch_price_history([benchmark], period=period, use_cache=use_cache)


def clear_cache() -> None:
    """Drop the in-process price cache."""
    _CACHE.clear()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _download_adjusted_close(tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=True,     # 'Close' is returned already adjusted
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as exc:
        raise MarketDataError(f"yfinance download failed: {exc}") from exc
    return _extract_close(raw, tickers)


def _extract_close(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Pull the (adjusted) Close column(s) out of a yfinance frame."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=tickers)

    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0)
        field_name = "Close" if "Close" in fields else ("Adj Close" if "Adj Close" in fields else None)
        if field_name is None:
            return pd.DataFrame(columns=tickers)
        close = raw[field_name].copy()
        if isinstance(close, pd.Series):
            close = close.to_frame()
    else:
        col = "Close" if "Close" in raw.columns else ("Adj Close" if "Adj Close" in raw.columns else None)
        if col is None:
            return pd.DataFrame(columns=tickers)
        close = raw[[col]].copy()
        close.columns = [tickers[0]]

    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def _unique_upper(tickers: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for t in tickers or []:
        s = str(t).strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out
