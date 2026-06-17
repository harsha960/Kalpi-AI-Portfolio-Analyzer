"""Offline unit tests for backend.core.simulation (no yfinance / network)."""
import numpy as np
import pandas as pd
import pytest

from backend.core import simulation as sim
from backend.core.simulation import SimulationError
from backend.models.schemas import Holding, Portfolio


def make_portfolio() -> Portfolio:
    return Portfolio(holdings=[
        Holding(ticker="RELIANCE.NS", weight=0.4),
        Holding(ticker="TCS.NS", weight=0.35),
        Holding(ticker="INFY.NS", weight=0.25),
    ])


def make_prices(cols) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame({c: np.linspace(100, 104, 5) for c in cols}, index=idx)


def test_exit_into_existing():
    after = sim.exit_and_reallocate(make_portfolio(), "RELIANCE.NS", "TCS.NS")
    wm = after.weight_map()
    assert "RELIANCE.NS" not in wm                      # exited
    assert wm["TCS.NS"] == pytest.approx(0.75)          # 0.35 + 0.40
    assert wm["INFY.NS"] == pytest.approx(0.25)
    assert sum(wm.values()) == pytest.approx(1.0)


def test_exit_into_new():
    after = sim.exit_and_reallocate(make_portfolio(), "INFY.NS", "GOLDBEES.NS")
    wm = after.weight_map()
    assert "INFY.NS" not in wm
    assert wm["GOLDBEES.NS"] == pytest.approx(0.25)     # new holding
    assert wm["RELIANCE.NS"] == pytest.approx(0.4)
    assert sum(wm.values()) == pytest.approx(1.0)


def test_invalid_exit_raises():
    with pytest.raises(SimulationError):
        sim.exit_and_reallocate(make_portfolio(), "WIPRO.NS", "TCS.NS")


def test_apply_weight_change_sum_one():
    after = sim.apply_weight_change(make_portfolio(), {"RELIANCE.NS": 0.6})
    wm = after.weight_map()
    assert sum(wm.values()) == pytest.approx(1.0)
    assert wm["RELIANCE.NS"] == pytest.approx(0.5)      # 0.6 / (0.6+0.35+0.25)


def test_apply_weight_change_zero_removes():
    after = sim.apply_weight_change(make_portfolio(), {"INFY.NS": 0.0})
    wm = after.weight_map()
    assert "INFY.NS" not in wm
    assert sum(wm.values()) == pytest.approx(1.0)


def test_simulate_what_if_explicit():
    prices = make_prices(["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    res = sim.simulate_what_if(make_portfolio(), prices,
                               exit_ticker="RELIANCE.NS", target_ticker="TCS.NS")
    assert res["changed"] is True
    assert "RELIANCE.NS" not in res["after_portfolio"]
    assert res["after_portfolio"]["TCS.NS"] == pytest.approx(0.75)
    assert sum(res["after_portfolio"].values()) == pytest.approx(1.0)
    assert sum(res["before_portfolio"].values()) == pytest.approx(1.0)


def test_simulate_what_if_parses_instruction():
    prices = make_prices(["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    res = sim.simulate_what_if(make_portfolio(), prices,
                               instruction="What if I exit Reliance and allocate it to TCS?")
    assert res["changed"] is True
    assert res["exit_ticker"] == "RELIANCE.NS"
    assert res["target_ticker"] == "TCS.NS"


def test_simulate_what_if_missing_target_price_warns():
    prices = make_prices(["RELIANCE.NS", "TCS.NS", "INFY.NS"])     # no GOLDBEES
    res = sim.simulate_what_if(make_portfolio(), prices,
                               exit_ticker="INFY.NS", target_ticker="GOLDBEES.NS")
    assert res["changed"] is True
    assert "GOLDBEES.NS" in res["after_portfolio"]
    assert "no price history" in res["note"].lower()


def test_simulate_what_if_unactionable():
    prices = make_prices(["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    res = sim.simulate_what_if(make_portfolio(), prices, instruction="tell me about my holdings")
    assert res["changed"] is False
    assert sum(res["after_portfolio"].values()) == pytest.approx(1.0)


def test_what_if_largest_to_gold():
    prices = make_prices(["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    res = sim.simulate_what_if(make_portfolio(), prices,
                               instruction="What if I exit my largest position and move it to gold?")
    assert res["changed"] is True
    assert res["exit_ticker"] == "RELIANCE.NS"        # largest weight (0.40)
    assert res["target_ticker"] == "GOLDBEES.NS"
    assert "RELIANCE.NS" not in res["after_portfolio"]
    assert "GOLDBEES.NS" in res["after_portfolio"]


def test_what_if_partial_move_asks_confirmation():
    port = Portfolio(holdings=[Holding(ticker="TCS.NS", weight=0.5),
                               Holding(ticker="HDFCBANK.NS", weight=0.5)])
    prices = make_prices(["TCS.NS", "HDFCBANK.NS"])
    res = sim.simulate_what_if(port, prices,
                               instruction="What if I reduce TCS by 10% and add it to HDFCBANK?")
    assert res["changed"] is False
    assert "confirm how much weight" in res["note"].lower()
    assert "TCS.NS" in res["note"] and "HDFCBANK.NS" in res["note"]
