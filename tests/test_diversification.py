"""Offline unit tests for backend.core.diversification (no yfinance / network)."""
import json

import numpy as np
import pandas as pd
import pytest

from backend.core import diversification as dv
from backend.core.diversification import DiversificationError
from backend.models.schemas import Holding, Portfolio


def _prices_from_returns(returns: pd.DataFrame, start: float = 100.0) -> pd.DataFrame:
    prices = start * (1.0 + returns).cumprod()
    init_idx = returns.index[0] - pd.Timedelta(days=1)
    init = pd.DataFrame({c: [start] for c in returns.columns}, index=[init_idx])
    return pd.concat([init, prices])


@pytest.fixture
def prices() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=6)
    a = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.015])
    returns = pd.DataFrame({"AAA": a, "BBB": a, "CCC": -a}, index=idx)  # B==A, C==-A
    return _prices_from_returns(returns)


SECTOR_MAP = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"}


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
def test_correlation_matrix_structure(prices):
    corr = dv.correlation_matrix(prices)
    assert list(corr.columns) == ["AAA", "BBB", "CCC"]
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr.to_numpy()), [1, 1, 1], atol=1e-12)
    assert corr.loc["AAA", "BBB"] == pytest.approx(1.0, abs=1e-9)    # identical
    assert corr.loc["AAA", "CCC"] == pytest.approx(-1.0, abs=1e-9)   # opposite


def test_weighted_correlation_score_equal_weights(prices):
    # pairs: AB=+1, AC=-1, BC=-1; equal weights -> (1-1-1)/3 = -1/3
    score = dv.weighted_correlation_score(prices, {"AAA": 1, "BBB": 1, "CCC": 1})
    assert score == pytest.approx(-1.0 / 3.0, abs=1e-9)


def test_weighted_correlation_mismatch_raises(prices):
    with pytest.raises(DiversificationError):
        dv.weighted_correlation_score(prices, {"AAA": 1.0})  # BBB, CCC missing


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #
def test_concentration_metrics_basic():
    m = dv.concentration_metrics({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    assert m["max_weight"] == pytest.approx(0.5)
    assert m["top_3_weight"] == pytest.approx(1.0)
    assert m["hhi"] == pytest.approx(0.5**2 + 0.3**2 + 0.2**2)
    assert m["largest_position"] == "AAA"
    assert m["num_holdings"] == 3


def test_concentration_normalizes_weights():
    # 50/30/20 (sum 100) must match 0.5/0.3/0.2
    m = dv.concentration_metrics({"AAA": 50, "BBB": 30, "CCC": 20})
    assert m["max_weight"] == pytest.approx(0.5)
    assert m["hhi"] == pytest.approx(0.38)


def test_concentration_top3_with_four_holdings():
    m = dv.concentration_metrics({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
    assert m["top_3_weight"] == pytest.approx(0.9)
    assert m["largest_position"] == "A"


def test_concentration_empty_raises():
    with pytest.raises(DiversificationError):
        dv.concentration_metrics({})


def test_concentration_negative_raises():
    with pytest.raises(DiversificationError):
        dv.concentration_metrics({"AAA": -0.5, "BBB": 1.5})


# --------------------------------------------------------------------------- #
# Sector exposure
# --------------------------------------------------------------------------- #
def test_sector_exposure_with_map():
    port = Portfolio(holdings=[Holding(ticker="AAA", weight=0.5),
                               Holding(ticker="BBB", weight=0.3),
                               Holding(ticker="CCC", weight=0.2)])
    exp = dv.sector_exposure(port, sector_map=SECTOR_MAP)
    assert exp["Tech"] == pytest.approx(0.8)
    assert exp["Energy"] == pytest.approx(0.2)


def test_sector_exposure_fallback_unknown():
    port = Portfolio(holdings=[Holding(ticker="AAA", weight=0.5),
                               Holding(ticker="BBB", weight=0.3),
                               Holding(ticker="CCC", weight=0.2)])
    exp = dv.sector_exposure(port, sector_map={"AAA": "Tech"})  # BBB, CCC unmapped
    assert exp["Tech"] == pytest.approx(0.5)
    assert exp["Unknown"] == pytest.approx(0.5)


def test_sector_exposure_default_all_unknown_offline():
    # No sector_map and lookup defaults to False -> no network, all Unknown.
    exp = dv.sector_exposure({"AAA": 0.5, "BBB": 0.5})
    assert exp == {"Unknown": 1.0}


def test_sector_exposure_none_raises():
    with pytest.raises(DiversificationError):
        dv.sector_exposure(None)


# --------------------------------------------------------------------------- #
# Summary + JSON-serializability
# --------------------------------------------------------------------------- #
def test_diversification_summary(prices):
    weights = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    summary = dv.diversification_summary(prices, weights, sector_map=SECTOR_MAP)

    assert set(summary) == {
        "correlation_matrix", "weighted_avg_correlation",
        "concentration", "sector_exposure",
    }
    assert summary["correlation_matrix"]["AAA"]["BBB"] == pytest.approx(1.0, abs=1e-9)
    assert summary["concentration"]["largest_position"] == "AAA"
    assert summary["sector_exposure"]["Tech"] == pytest.approx(0.8)

    # The entire summary must be JSON-serializable.
    json.dumps(summary)


def test_empty_prices_raises():
    with pytest.raises(DiversificationError):
        dv.correlation_matrix(pd.DataFrame())
