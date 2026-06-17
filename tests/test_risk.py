"""Offline unit tests for backend.core.risk (no yfinance / network)."""
import math

import numpy as np
import pandas as pd
import pytest

from backend.core import risk
from backend.core.risk import RiskError


@pytest.fixture
def returns() -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=10)
    vals = [0.02, -0.05, 0.03, -0.01, 0.04, -0.02, 0.01, -0.03, 0.02, -0.04]
    return pd.Series(vals, index=idx, name="portfolio")


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
def test_drawdown_series_formula(returns):
    cumulative = (1.0 + returns).cumprod()
    expected = (cumulative / cumulative.cummax() - 1.0).rename("drawdown")
    pd.testing.assert_series_equal(risk.drawdown_series(returns), expected)


def test_maximum_drawdown_simple():
    # cumulative [1.1, 0.55] -> min drawdown = 0.55/1.1 - 1 = -0.5
    assert risk.maximum_drawdown(pd.Series([0.1, -0.5])) == pytest.approx(-0.5, rel=1e-12)


def test_maximum_drawdown_matches_series(returns):
    assert risk.maximum_drawdown(returns) == pytest.approx(
        risk.drawdown_series(returns).min(), rel=1e-12
    )


def test_no_drawdown_when_monotonic():
    assert risk.maximum_drawdown(pd.Series([0.01, 0.02, 0.03])) == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------- #
# VaR / CVaR
# --------------------------------------------------------------------------- #
def test_historical_var_95(returns):
    expected = float(np.quantile(returns.to_numpy(), 0.05))
    assert risk.historical_var(returns) == pytest.approx(expected, rel=1e-12)


def test_historical_var_custom_confidence(returns):
    expected = float(np.quantile(returns.to_numpy(), 0.01))
    assert risk.historical_var(returns, confidence_level=0.99) == pytest.approx(expected, rel=1e-12)


def test_historical_cvar_95(returns):
    var = float(np.quantile(returns.to_numpy(), 0.05))
    tail = returns[returns <= var]
    assert risk.historical_cvar(returns) == pytest.approx(float(tail.mean()), rel=1e-12)


def test_cvar_not_above_var(returns):
    assert risk.historical_cvar(returns) <= risk.historical_var(returns) + 1e-15


# --------------------------------------------------------------------------- #
# Downside deviation
# --------------------------------------------------------------------------- #
def test_downside_deviation(returns):
    neg = returns[returns < 0.0]
    expected = neg.std(ddof=1) * math.sqrt(252)
    assert risk.downside_deviation(returns) == pytest.approx(expected, rel=1e-12)


def test_downside_deviation_no_negatives():
    assert risk.downside_deviation(pd.Series([0.01, 0.02, 0.0, 0.03])) == 0.0


# --------------------------------------------------------------------------- #
# Summary + errors
# --------------------------------------------------------------------------- #
def test_risk_summary_keys_and_values(returns):
    s = risk.risk_summary(returns)
    assert set(s) == {
        "annualized_volatility", "max_drawdown",
        "historical_var_95", "historical_cvar_95", "downside_deviation",
    }
    assert s["max_drawdown"] == pytest.approx(risk.maximum_drawdown(returns), rel=1e-12)
    assert s["historical_var_95"] == pytest.approx(risk.historical_var(returns), rel=1e-12)
    assert s["historical_cvar_95"] == pytest.approx(risk.historical_cvar(returns), rel=1e-12)
    assert s["downside_deviation"] == pytest.approx(risk.downside_deviation(returns), rel=1e-12)

    from backend.core.performance import annualized_volatility
    assert s["annualized_volatility"] == pytest.approx(annualized_volatility(returns), rel=1e-12)


def test_empty_series_raises():
    with pytest.raises(RiskError):
        risk.drawdown_series(pd.Series([], dtype=float))


def test_risk_summary_single_obs_raises():
    with pytest.raises(RiskError):
        risk.risk_summary(pd.Series([0.01]))


def test_invalid_confidence_raises(returns):
    with pytest.raises(RiskError):
        risk.historical_var(returns, confidence_level=1.5)
