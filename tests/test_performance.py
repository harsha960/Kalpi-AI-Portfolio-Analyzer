"""Offline unit tests for backend.core.performance.

All price data is synthetic (built from chosen return paths), so these tests
never touch yfinance or the network. Each metric is checked against an
independent numpy/pandas reference computation.
"""
import math

import numpy as np
import pandas as pd
import pytest

from backend.core import performance as perf

WEIGHTS = {"AAA": 0.6, "BBB": 0.4}


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _prices_from_returns(returns: pd.DataFrame, start: float = 100.0) -> pd.DataFrame:
    """Build prices whose pct_change().dropna() reproduces `returns` exactly."""
    prices = start * (1.0 + returns).cumprod()
    init_idx = returns.index[0] - pd.Timedelta(days=1)
    init = pd.DataFrame({c: [start] for c in returns.columns}, index=[init_idx])
    return pd.concat([init, prices])


@pytest.fixture
def returns_df() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame(
        {
            "AAA": [0.01, -0.02, 0.015, 0.00, 0.005],
            "BBB": [0.00, 0.01, -0.005, 0.02, -0.01],
        },
        index=idx,
    )


@pytest.fixture
def prices_df(returns_df) -> pd.DataFrame:
    return _prices_from_returns(returns_df)


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def test_daily_returns_recovers_inputs(prices_df, returns_df):
    out = perf.calculate_daily_returns(prices_df)
    assert list(out.columns) == ["AAA", "BBB"]
    assert len(out) == 5
    pd.testing.assert_frame_equal(out, returns_df, check_freq=False)


def test_portfolio_returns_weighted_sum(prices_df, returns_df):
    out = perf.calculate_portfolio_returns(prices_df, WEIGHTS)
    expected = (returns_df * pd.Series(WEIGHTS)).sum(axis=1)
    assert isinstance(out, pd.Series)
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-9, atol=1e-12)


def test_portfolio_returns_accepts_list_weights(prices_df, returns_df):
    out = perf.calculate_portfolio_returns(prices_df, [0.6, 0.4])
    expected = (returns_df * np.array([0.6, 0.4])).sum(axis=1)
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-9)


def test_weights_are_renormalized(prices_df, returns_df):
    # 6/4 should behave identically to 0.6/0.4 after internal normalization
    out = perf.calculate_portfolio_returns(prices_df, {"AAA": 6, "BBB": 4})
    expected = (returns_df * pd.Series({"AAA": 0.6, "BBB": 0.4})).sum(axis=1)
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-9)


def test_cumulative_returns(prices_df):
    port = perf.calculate_portfolio_returns(prices_df, WEIGHTS)
    cum = perf.calculate_cumulative_returns(port)
    assert len(cum) == len(port)
    assert cum.iloc[-1] == pytest.approx((1.0 + port).prod() - 1.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# Scalar metrics
# --------------------------------------------------------------------------- #
def test_annualized_return(prices_df):
    port = perf.calculate_portfolio_returns(prices_df, WEIGHTS)
    n = len(port)
    total = (1.0 + port).prod() - 1.0
    expected = (1.0 + total) ** (252 / n) - 1.0
    assert perf.annualized_return(port) == pytest.approx(expected, rel=1e-12)


def test_annualized_volatility(prices_df):
    port = perf.calculate_portfolio_returns(prices_df, WEIGHTS)
    expected = port.std(ddof=1) * math.sqrt(252)
    assert perf.annualized_volatility(port) == pytest.approx(expected, rel=1e-12)


def test_sharpe_ratio(prices_df):
    port = perf.calculate_portfolio_returns(prices_df, WEIGHTS)
    ann_ret = perf.annualized_return(port)
    ann_vol = perf.annualized_volatility(port)
    assert perf.sharpe_ratio(port) == pytest.approx((ann_ret - 0.065) / ann_vol, rel=1e-12)
    assert perf.sharpe_ratio(port, risk_free_rate=0.0) == pytest.approx(ann_ret / ann_vol, rel=1e-12)


# --------------------------------------------------------------------------- #
# Benchmark comparison + date alignment
# --------------------------------------------------------------------------- #
def test_benchmark_comparison_aligns_dates(returns_df):
    pf_prices = _prices_from_returns(returns_df)

    bm_idx = pd.bdate_range("2024-01-03", periods=6)  # shifted/extends beyond portfolio
    bm_returns = pd.DataFrame(
        {"^NSEI": [0.002, -0.001, 0.004, 0.0, 0.003, -0.002]}, index=bm_idx
    )
    bm_prices = _prices_from_returns(bm_returns)

    result = perf.benchmark_comparison(pf_prices, bm_prices, WEIGHTS)

    common = pf_prices.index.intersection(bm_prices.index)
    assert result["trading_days"] == len(common) - 1
    assert result["start"] == str(common.min().date())
    assert result["end"] == str(common.max().date())

    pf_ret = perf.calculate_portfolio_returns(pf_prices.loc[common].sort_index(), WEIGHTS)
    assert result["portfolio"]["annualized_return"] == pytest.approx(
        perf.annualized_return(pf_ret), rel=1e-12
    )
    assert result["portfolio"]["sharpe_ratio"] == pytest.approx(
        perf.sharpe_ratio(pf_ret), rel=1e-12
    )

    bm_ret = bm_prices.loc[common].sort_index().iloc[:, 0].pct_change().dropna()
    assert result["benchmark"]["annualized_volatility"] == pytest.approx(
        perf.annualized_volatility(bm_ret), rel=1e-12
    )
    assert result["benchmark"]["name"] == "^NSEI"
    assert result["outperformed"] == (
        result["portfolio"]["annualized_return"] > result["benchmark"]["annualized_return"]
    )


def test_performance_summary(prices_df):
    s = perf.performance_summary(prices_df, WEIGHTS)
    assert set(s) >= {
        "total_return", "annualized_return", "annualized_volatility",
        "sharpe_ratio", "observations",
    }
    port = perf.calculate_portfolio_returns(prices_df, WEIGHTS)
    assert s["sharpe_ratio"] == pytest.approx(perf.sharpe_ratio(port), rel=1e-12)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def test_empty_prices_raise():
    with pytest.raises(perf.PerformanceError):
        perf.calculate_daily_returns(pd.DataFrame())


def test_single_row_volatility_raises():
    with pytest.raises(perf.PerformanceError):
        perf.annualized_volatility(pd.Series([0.01]))


def test_weight_length_mismatch_raises(prices_df):
    with pytest.raises(perf.PerformanceError):
        perf.calculate_portfolio_returns(prices_df, [1.0])  # 1 weight, 2 columns


def test_missing_weight_for_column_raises(prices_df):
    with pytest.raises(perf.PerformanceError):
        perf.calculate_portfolio_returns(prices_df, {"AAA": 1.0})  # BBB missing
