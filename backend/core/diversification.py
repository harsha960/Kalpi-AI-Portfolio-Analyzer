"""Deterministic diversification & concentration analysis.

Pure pandas/numpy math — no LLM, no simulation, no API/UI. The only optional
network touch is a *safe, opt-in* yfinance sector lookup (``lookup=True``);
it is never triggered by default, so this module is offline by default.
All summary outputs are JSON-serializable.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..models.schemas import Portfolio

__all__ = [
    "DiversificationError",
    "correlation_matrix",
    "weighted_correlation_score",
    "concentration_metrics",
    "sector_exposure",
    "diversification_summary",
]


class DiversificationError(ValueError):
    """Raised when diversification metrics cannot be computed from the inputs."""


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
def correlation_matrix(prices) -> pd.DataFrame:
    """Pearson correlation of daily returns: ``prices.pct_change().dropna().corr()``."""
    df = _as_price_frame(prices)
    returns = df.pct_change().dropna()
    if returns.empty:
        raise DiversificationError(
            "Not enough price data to compute correlations (need >= 2 dated rows)."
        )
    return returns.corr()


def weighted_correlation_score(prices, weights) -> float:
    """Weight-weighted average of the off-diagonal pairwise correlations.

    Each pair (i, j), i != j, is weighted by ``w_i * w_j``; the diagonal
    (self-correlation) is ignored. Returns 0.0 when there are fewer than two
    assets or the weights are fully concentrated in one name.
    """
    corr = correlation_matrix(prices)
    tickers = list(corr.columns)
    n = len(tickers)
    if n < 2:
        return 0.0

    w = _align_weights_to(tickers, weights).reindex(tickers).to_numpy(dtype=float)
    C = corr.to_numpy(dtype=float)
    W = np.outer(w, w)

    off_diag = ~np.eye(n, dtype=bool)
    finite = off_diag & np.isfinite(C)        # skip undefined correlations
    den = float(W[finite].sum())
    if den <= 0.0:
        return 0.0
    num = float((W * C)[finite].sum())
    return num / den


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #
def concentration_metrics(weights) -> Dict[str, object]:
    """Concentration stats from normalized weights (max, top-3, HHI, largest)."""
    w = _normalized_weights(weights).sort_values(ascending=False)
    return {
        "max_weight": float(w.iloc[0]),
        "top_3_weight": float(w.iloc[:3].sum()),
        "hhi": float((w ** 2).sum()),
        "largest_position": str(w.index[0]),
        "num_holdings": int(len(w)),
        "weights": {str(k): float(v) for k, v in w.items()},
    }


# --------------------------------------------------------------------------- #
# Sector exposure
# --------------------------------------------------------------------------- #
def sector_exposure(portfolio, sector_map: Optional[Mapping] = None, *, lookup: bool = False) -> Dict[str, float]:
    """Aggregate normalized weights by sector.

    ``sector_map`` (ticker -> sector) takes precedence. If it is missing a
    ticker and ``lookup`` is True, a safe yfinance lookup is attempted; any
    unresolved ticker falls back to the "Unknown" sector.
    """
    weights = _portfolio_to_weights(portfolio)
    smap = dict(sector_map) if sector_map else {}

    exposures: Dict[str, float] = {}
    for ticker, weight in weights.items():
        sector = smap.get(ticker)
        if sector is None and lookup:
            sector = _lookup_sector_safe(ticker)
        if not sector:
            sector = "Unknown"
        exposures[sector] = exposures.get(sector, 0.0) + float(weight)

    # Largest exposure first, for readable/JSON-friendly output.
    return dict(sorted(exposures.items(), key=lambda kv: kv[1], reverse=True))


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def diversification_summary(
    prices,
    weights,
    portfolio: Optional[Portfolio] = None,
    sector_map: Optional[Mapping] = None,
    *,
    lookup: bool = False,
) -> Dict[str, object]:
    """Bundle correlation, weighted-avg correlation, concentration, and sectors."""
    corr = correlation_matrix(prices)
    score = weighted_correlation_score(prices, weights)
    concentration = concentration_metrics(weights)
    holdings_source = portfolio if portfolio is not None else weights
    sectors = sector_exposure(holdings_source, sector_map=sector_map, lookup=lookup)

    return {
        "correlation_matrix": _corr_to_jsonable(corr),
        "weighted_avg_correlation": float(score),
        "concentration": concentration,
        "sector_exposure": sectors,
    }


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _as_price_frame(prices) -> pd.DataFrame:
    if prices is None:
        raise DiversificationError("No price data provided.")
    if hasattr(prices, "prices") and isinstance(getattr(prices, "prices"), pd.DataFrame):
        prices = prices.prices
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    if not isinstance(prices, pd.DataFrame):
        raise DiversificationError(f"prices must be a pandas DataFrame, got {type(prices).__name__}.")
    if prices.empty or prices.shape[1] == 0:
        raise DiversificationError("Price data is empty.")
    return prices.sort_index()


def _weights_to_series(weights) -> pd.Series:
    if weights is None:
        raise DiversificationError("No weights provided.")
    if hasattr(weights, "weight_map") and callable(getattr(weights, "weight_map")):
        weights = weights.weight_map()                      # a Portfolio
    if isinstance(weights, pd.Series):
        s = weights.copy()
    elif isinstance(weights, Mapping):
        s = pd.Series(dict(weights))
    else:
        try:
            values = list(weights)
        except TypeError:
            raise DiversificationError(f"Unsupported weights type: {type(weights).__name__}.")
        s = pd.Series(values, index=[str(i) for i in range(len(values))])
    s.index = s.index.map(str)
    return s


def _normalized_weights(weights) -> pd.Series:
    s = _weights_to_series(weights).dropna().astype(float)
    if s.empty:
        raise DiversificationError("No weights provided.")
    if (s < 0).any():
        raise DiversificationError("Weights must be non-negative.")
    total = float(s.sum())
    if total <= 0:
        raise DiversificationError("Sum of weights must be positive.")
    return s / total


def _align_weights_to(tickers, weights) -> pd.Series:
    cols = [str(t) for t in tickers]
    s = _weights_to_series(weights)
    aligned = s.reindex(cols)
    if aligned.isna().any():
        missing = [c for c in cols if c not in s.index or pd.isna(s.get(c))]
        raise DiversificationError(f"No weight provided for ticker(s): {missing}.")
    aligned = aligned.astype(float)
    if (aligned < 0).any():
        raise DiversificationError("Weights must be non-negative.")
    total = float(aligned.sum())
    if total <= 0:
        raise DiversificationError("Sum of aligned weights must be positive.")
    return aligned / total


def _portfolio_to_weights(portfolio) -> pd.Series:
    if portfolio is None:
        raise DiversificationError("No portfolio provided.")
    if hasattr(portfolio, "weight_map") and callable(getattr(portfolio, "weight_map")):
        s = pd.Series(portfolio.weight_map())
    elif isinstance(portfolio, Mapping):
        s = pd.Series(dict(portfolio))
    elif isinstance(portfolio, pd.Series):
        s = portfolio.copy()
    else:
        raise DiversificationError(
            "portfolio must be a Portfolio, dict, or Series of {ticker: weight}."
        )
    s.index = s.index.map(str)
    s = s.dropna().astype(float)
    if s.empty:
        raise DiversificationError("Portfolio has no holdings.")
    if (s < 0).any():
        raise DiversificationError("Weights must be non-negative.")
    total = float(s.sum())
    if total <= 0:
        raise DiversificationError("Sum of weights must be positive.")
    return s / total


def _lookup_sector_safe(ticker: str) -> Optional[str]:
    """Best-effort yfinance sector lookup; returns None on any failure."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        sector = info.get("sector")
        return str(sector) if sector else None
    except Exception:
        return None


def _corr_to_jsonable(corr: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    return {
        str(col): {
            str(idx): (None if pd.isna(val) else float(val))
            for idx, val in corr[col].items()
        }
        for col in corr.columns
    }
