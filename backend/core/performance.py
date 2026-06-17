"""Deterministic performance metrics.

Pure pandas/numpy math — no LLM, no market-data calls, and no risk or
diversification logic. Every result here is exact and reproducible; the agent
only *calls* these functions and formats their numbers (the "Golden Rule").

Conventions
-----------
* ``prices``  : pandas DataFrame, index = dates, columns = tickers.
* returns     : daily simple returns = ``prices.pct_change().dropna()``.
* ``weights`` : a Portfolio, dict {ticker: w}, pandas Series, or a positional
                sequence aligned to the price columns. Weights are aligned to
                the available columns and re-scaled to sum to 1.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Dict, Union

import numpy as np
import pandas as pd

__all__ = [
    "PerformanceError",
    "calculate_daily_returns",
    "calculate_portfolio_returns",
    "calculate_cumulative_returns",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "benchmark_comparison",
    "performance_summary",
]

Weights = Union[Mapping, pd.Series, list, tuple, np.ndarray, object]


class PerformanceError(ValueError):
    """Raised when performance metrics cannot be computed from the given data."""


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def calculate_daily_returns(prices) -> pd.DataFrame:
    """Daily simple returns: ``prices.pct_change().dropna()``."""
    df = _as_price_frame(prices)
    returns = df.pct_change().dropna()
    if returns.empty:
        raise PerformanceError(
            "Not enough price data to compute daily returns (need >= 2 dated rows)."
        )
    return returns


def calculate_portfolio_returns(prices, weights: Weights) -> pd.Series:
    """Weighted sum of each asset's daily returns -> a portfolio return series."""
    daily = calculate_daily_returns(prices)
    w = _align_weights(daily.columns, weights)
    port = daily.mul(w, axis=1).sum(axis=1)
    port.name = "portfolio"
    return port


def calculate_cumulative_returns(portfolio_returns) -> pd.Series:
    """Compounded cumulative return series: ``(1 + r).cumprod() - 1``."""
    r = _as_return_series(portfolio_returns)
    cum = (1.0 + r).cumprod() - 1.0
    cum.name = "cumulative_return"
    return cum


# --------------------------------------------------------------------------- #
# Scalar metrics
# --------------------------------------------------------------------------- #
def annualized_return(portfolio_returns, trading_days: int = 252) -> float:
    """Annualized return from the compounded total return.

    ``(1 + total_return) ** (trading_days / number_of_days) - 1``
    """
    r = _as_return_series(portfolio_returns)
    n = len(r)
    if n == 0:
        raise PerformanceError("Cannot annualize: empty return series.")
    total_return = float((1.0 + r).prod() - 1.0)
    base = 1.0 + total_return
    if base <= 0.0:                      # total wipeout (or worse)
        return -1.0
    return base ** (trading_days / n) - 1.0


def annualized_volatility(portfolio_returns, trading_days: int = 252) -> float:
    """Annualized volatility: ``daily_std * sqrt(trading_days)`` (sample std)."""
    r = _as_return_series(portfolio_returns)
    if len(r) < 2:
        raise PerformanceError("Need >= 2 return observations to compute volatility.")
    daily_std = float(r.std(ddof=1))     # sample standard deviation
    return daily_std * math.sqrt(trading_days)


def sharpe_ratio(
    portfolio_returns,
    risk_free_rate: float = 0.065,
    trading_days: int = 252,
) -> float:
    """``(annualized_return - risk_free_rate) / annualized_volatility``."""
    r = _as_return_series(portfolio_returns)
    ann_ret = annualized_return(r, trading_days)
    ann_vol = annualized_volatility(r, trading_days)
    if ann_vol == 0.0:
        raise PerformanceError("Annualized volatility is zero; Sharpe ratio is undefined.")
    return (ann_ret - risk_free_rate) / ann_vol


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def benchmark_comparison(
    portfolio_prices,
    benchmark_prices,
    weights: Weights,
    *,
    risk_free_rate: float = 0.065,
    trading_days: int = 252,
) -> Dict[str, object]:
    """Compare the portfolio against a benchmark over their common dates."""
    pf_prices = _as_price_frame(portfolio_prices)
    bm_series = _as_benchmark_series(benchmark_prices)

    common = pf_prices.index.intersection(bm_series.index)
    if len(common) < 2:
        raise PerformanceError("Portfolio and benchmark share fewer than 2 common dates.")

    pf_aligned = pf_prices.loc[common].sort_index()
    bm_aligned = bm_series.loc[common].sort_index()

    pf_returns = calculate_portfolio_returns(pf_aligned, weights)
    bm_returns = bm_aligned.pct_change().dropna()
    bm_returns.name = "benchmark"

    pf_block = _metric_block(pf_returns, risk_free_rate, trading_days)
    bm_block = _metric_block(bm_returns, risk_free_rate, trading_days)
    bm_block["name"] = _series_name(benchmark_prices, default="benchmark")

    excess = pf_block["annualized_return"] - bm_block["annualized_return"]
    return {
        "start": _date_str(common.min()),
        "end": _date_str(common.max()),
        "trading_days": int(len(pf_returns)),
        "risk_free_rate": float(risk_free_rate),
        "portfolio": pf_block,
        "benchmark": bm_block,
        "excess_annualized_return": float(excess),
        "outperformed": bool(pf_block["annualized_return"] > bm_block["annualized_return"]),
    }


def performance_summary(
    prices,
    weights: Weights,
    *,
    risk_free_rate: float = 0.065,
    trading_days: int = 252,
) -> Dict[str, float]:
    """Convenience: total/annualized return, volatility, and Sharpe in one dict."""
    port = calculate_portfolio_returns(prices, weights)
    block = _metric_block(port, risk_free_rate, trading_days)
    block["observations"] = int(len(port))
    return block


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _metric_block(returns: pd.Series, risk_free_rate: float, trading_days: int) -> Dict[str, float]:
    return {
        "total_return": float((1.0 + returns).prod() - 1.0),
        "annualized_return": float(annualized_return(returns, trading_days)),
        "annualized_volatility": float(annualized_volatility(returns, trading_days)),
        "sharpe_ratio": float(sharpe_ratio(returns, risk_free_rate, trading_days)),
    }


def _as_price_frame(prices) -> pd.DataFrame:
    if prices is None:
        raise PerformanceError("No price data provided.")
    if hasattr(prices, "prices") and isinstance(getattr(prices, "prices"), pd.DataFrame):
        prices = prices.prices                       # accept a PriceHistory dataclass
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    if not isinstance(prices, pd.DataFrame):
        raise PerformanceError(f"prices must be a pandas DataFrame, got {type(prices).__name__}.")
    if prices.empty or prices.shape[1] == 0:
        raise PerformanceError("Price data is empty.")
    return prices.sort_index()


def _as_return_series(returns) -> pd.Series:
    if returns is None:
        raise PerformanceError("No return data provided.")
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 1:
            returns = returns.iloc[:, 0]
        else:
            raise PerformanceError("Expected a single return series, got multiple columns.")
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    returns = returns.dropna()
    if returns.empty:
        raise PerformanceError("Return series is empty.")
    return returns.astype(float)


def _align_weights(columns, weights) -> pd.Series:
    cols = list(columns)
    if hasattr(weights, "weight_map") and callable(getattr(weights, "weight_map")):
        weights = weights.weight_map()               # accept a Portfolio
    if isinstance(weights, pd.Series):
        w = weights.reindex(cols)
    elif isinstance(weights, Mapping):
        w = pd.Series({c: weights.get(c) for c in cols}, dtype="float64")
    else:
        arr = np.asarray(list(weights), dtype="float64")
        if arr.shape[0] != len(cols):
            raise PerformanceError(
                f"weights length {arr.shape[0]} does not match number of price columns {len(cols)}."
            )
        w = pd.Series(arr, index=cols)
    if w.isna().any():
        missing = [c for c in cols if pd.isna(w.get(c))]
        raise PerformanceError(f"No weight provided for column(s): {missing}.")
    w = w.astype(float)
    total = float(w.sum())
    if total <= 0.0:
        raise PerformanceError("Sum of weights must be positive.")
    return w / total


def _as_benchmark_series(benchmark_prices) -> pd.Series:
    if benchmark_prices is None:
        raise PerformanceError("No benchmark price data provided.")
    if hasattr(benchmark_prices, "prices") and isinstance(getattr(benchmark_prices, "prices"), pd.DataFrame):
        benchmark_prices = benchmark_prices.prices
    if isinstance(benchmark_prices, pd.DataFrame):
        if benchmark_prices.shape[1] == 0:
            raise PerformanceError("Benchmark price data is empty.")
        series = benchmark_prices.iloc[:, 0]
    elif isinstance(benchmark_prices, pd.Series):
        series = benchmark_prices
    else:
        raise PerformanceError("benchmark_prices must be a pandas DataFrame or Series.")
    series = series.dropna()
    if series.empty:
        raise PerformanceError("Benchmark price series is empty.")
    return series.sort_index()


def _series_name(benchmark_prices, default: str = "benchmark") -> str:
    if hasattr(benchmark_prices, "prices"):
        benchmark_prices = benchmark_prices.prices
    if isinstance(benchmark_prices, pd.DataFrame) and benchmark_prices.shape[1] >= 1:
        return str(benchmark_prices.columns[0])
    if isinstance(benchmark_prices, pd.Series) and benchmark_prices.name:
        return str(benchmark_prices.name)
    return default


def _date_str(value) -> str:
    try:
        return str(pd.Timestamp(value).date())
    except Exception:
        return str(value)
