"""Deterministic risk metrics.

Pure pandas/numpy math — no LLM, no market-data calls, and no diversification
or simulation logic. The agent only *calls* these functions and formats their
numbers (the project's "Golden Rule").

Sign convention: VaR and CVaR are reported as the (typically negative) tail
*returns* themselves, e.g. a VaR of -0.045 means "on the worst 5% of days the
return was about -4.5% or worse".
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

from .performance import annualized_volatility

__all__ = [
    "RiskError",
    "drawdown_series",
    "maximum_drawdown",
    "historical_var",
    "historical_cvar",
    "downside_deviation",
    "risk_summary",
]


class RiskError(ValueError):
    """Raised when risk metrics cannot be computed from the given data."""


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
def drawdown_series(portfolio_returns) -> pd.Series:
    """Drawdown at each point: ``cumulative / running_max - 1``."""
    r = _as_return_series(portfolio_returns)
    cumulative = (1.0 + r).cumprod()
    running_max = cumulative.cummax()
    dd = cumulative / running_max - 1.0
    dd.name = "drawdown"
    return dd


def maximum_drawdown(portfolio_returns) -> float:
    """The deepest peak-to-trough loss: the minimum of the drawdown series."""
    return float(drawdown_series(portfolio_returns).min())


# --------------------------------------------------------------------------- #
# Value at Risk
# --------------------------------------------------------------------------- #
def historical_var(portfolio_returns, confidence_level: float = 0.95) -> float:
    """Historical VaR: the (1 - confidence) percentile daily return.

    At 95% confidence this is the 5th-percentile daily return (a left-tail,
    usually negative, number).
    """
    r = _as_return_series(portfolio_returns)
    _validate_confidence(confidence_level)
    alpha = 1.0 - confidence_level
    return float(np.quantile(r.to_numpy(), alpha))


def historical_cvar(portfolio_returns, confidence_level: float = 0.95) -> float:
    """Historical CVaR / Expected Shortfall: mean of returns at or below VaR."""
    r = _as_return_series(portfolio_returns)
    var = historical_var(r, confidence_level)
    tail = r[r <= var]
    if tail.empty:
        return var
    return float(tail.mean())


# --------------------------------------------------------------------------- #
# Downside deviation
# --------------------------------------------------------------------------- #
def downside_deviation(portfolio_returns, trading_days: int = 252) -> float:
    """Annualized standard deviation of the negative daily returns only.

    Returns 0.0 when there are fewer than two negative observations (no
    measurable downside dispersion).
    """
    r = _as_return_series(portfolio_returns)
    negative = r[r < 0.0]
    if len(negative) < 2:
        return 0.0
    daily = float(negative.std(ddof=1))     # sample standard deviation
    return daily * math.sqrt(trading_days)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def risk_summary(portfolio_returns, trading_days: int = 252) -> Dict[str, float]:
    """Bundle the core risk metrics into one dictionary (VaR/CVaR at 95%)."""
    r = _as_return_series(portfolio_returns)
    if len(r) < 2:
        raise RiskError("Need >= 2 return observations to compute a risk summary.")
    return {
        "annualized_volatility": float(annualized_volatility(r, trading_days)),
        "max_drawdown": maximum_drawdown(r),
        "historical_var_95": historical_var(r, 0.95),
        "historical_cvar_95": historical_cvar(r, 0.95),
        "downside_deviation": downside_deviation(r, trading_days),
    }


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _as_return_series(returns) -> pd.Series:
    if returns is None:
        raise RiskError("No return data provided.")
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 1:
            returns = returns.iloc[:, 0]
        else:
            raise RiskError("Expected a single return series, got multiple columns.")
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    returns = returns.dropna().astype(float)
    if returns.empty:
        raise RiskError("Return series is empty.")
    return returns


def _validate_confidence(confidence_level: float) -> None:
    if not (0.0 < confidence_level < 1.0):
        raise RiskError(f"confidence_level must be between 0 and 1, got {confidence_level}.")
