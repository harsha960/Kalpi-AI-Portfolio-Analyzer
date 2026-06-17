"""Offline unit tests for the agent layer: orchestrator + tools (no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from backend.agent import orchestrator as orch
from backend.agent import tools
from backend.models.schemas import Holding, Portfolio


# --------------------------------------------------------------------------- #
# Orchestrator: rule-based classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("message,expected", [
    ("How are my returns and performance versus the Nifty benchmark?", "performance"),
    ("Show my risk: volatility, drawdown and VaR", "risk"),
    ("What is my sector exposure and concentration?", "diversification"),
    ("Show me the correlation matrix of my holdings", "correlation"),
    ("What if I exit Reliance and allocate to gold?", "what_if"),
    ("Hello, what can you do?", "general"),
])
def test_rule_based_intents(message, expected, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = orch.classify_intent(message)
    assert result["intent"] == expected
    assert result["method"] == "rules"
    assert 0.0 <= result["confidence"] <= 1.0


def test_supported_intents_constant():
    assert set(orch.SUPPORTED_INTENTS) == {
        "performance", "risk", "diversification", "correlation",
        "what_if", "summary", "holding_lookup", "general",
    }


def test_what_if_priority_over_risk(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert orch.classify_intent("What if I rebalance to reduce volatility?")["intent"] == "what_if"


def test_empty_message_is_general(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert orch.classify_intent("")["intent"] == "general"


def test_use_llm_false_forces_rules():
    r = orch.classify_intent("returns vs nifty", use_llm=False)
    assert r["method"] == "rules"
    assert r["intent"] == "performance"


def _raise(*_args, **_kwargs):
    raise RuntimeError("simulated LLM failure")


def test_llm_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(orch, "_classify_with_llm", _raise)
    r = orch.classify_intent("returns vs nifty")
    assert r["method"] == "rules"
    assert r["intent"] == "performance"


def test_llm_success_path(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(orch, "_classify_with_llm", lambda _m: "risk")
    r = orch.classify_intent("anything at all")
    assert r["intent"] == "risk"
    assert r["method"] == "llm"


# --------------------------------------------------------------------------- #
# Tools: deterministic wrappers (offline fake market)
# --------------------------------------------------------------------------- #
def _prices_from_returns(returns: pd.DataFrame, start: float = 100.0) -> pd.DataFrame:
    prices = start * (1.0 + returns).cumprod()
    init = pd.DataFrame({c: [start] for c in returns.columns},
                        index=[returns.index[0] - pd.Timedelta(days=1)])
    return pd.concat([init, prices])


@pytest.fixture
def market():
    idx = pd.bdate_range("2024-01-01", periods=8)
    rng = np.random.default_rng(0)
    cols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    prices = _prices_from_returns(pd.DataFrame(rng.normal(0.001, 0.01, (8, 3)), index=idx, columns=cols))
    bench = _prices_from_returns(pd.DataFrame(rng.normal(0.0008, 0.008, (8, 1)), index=idx, columns=["^NSEI"]))
    port = Portfolio(holdings=[Holding(ticker="RELIANCE.NS", weight=0.4),
                               Holding(ticker="TCS.NS", weight=0.35),
                               Holding(ticker="INFY.NS", weight=0.25)])
    return port, prices, bench


def test_run_performance_analysis(market):
    port, prices, bench = market
    r = tools.run_performance_analysis(port, prices, benchmark_prices=bench)
    assert r["intent"] == "performance"
    assert "annualized_return" in r["metrics"]
    assert "benchmark" in r
    json.dumps(r)                          # must be JSON-serializable


def test_run_risk_analysis(market):
    port, prices, _ = market
    r = tools.run_risk_analysis(port, prices)
    assert {"annualized_volatility", "max_drawdown", "historical_var_95"} <= set(r["metrics"])
    json.dumps(r)


def test_run_diversification_analysis(market):
    port, prices, _ = market
    smap = {"RELIANCE.NS": "Energy", "TCS.NS": "IT", "INFY.NS": "IT"}
    r = tools.run_diversification_analysis(port, prices, sector_map=smap)
    assert "correlation_matrix" in r and "sector_exposure" in r
    assert r["sector_exposure"]["IT"] == pytest.approx(0.6)
    json.dumps(r)


def test_run_correlation_analysis(market):
    port, prices, _ = market
    r = tools.run_correlation_analysis(port, prices)
    assert r["intent"] == "correlation"
    assert set(r["correlation_matrix"]) == {"RELIANCE.NS", "TCS.NS", "INFY.NS"}
    json.dumps(r)


def test_run_what_if_analysis(market):
    port, prices, _ = market
    r = tools.run_what_if_analysis(port, prices, "What if I exit INFY and allocate to TCS?")
    assert r["intent"] == "what_if"
    assert r["simulation"]["changed"] is True
    assert "after_metrics" in r            # both resulting tickers have price data
    json.dumps(r)


def test_run_diversification_uses_default_sector_map(market):
    # No sector_map passed -> DEFAULT_SECTOR_MAP is used (offline, no yfinance).
    port, prices, _ = market
    r = tools.run_diversification_analysis(port, prices)
    sectors = r["sector_exposure"]
    assert "Unknown" not in sectors                       # sample tickers are mapped
    assert "Information Technology" in sectors
    assert sectors["Information Technology"] == pytest.approx(0.6)   # TCS 0.35 + INFY 0.25
    assert sectors["Energy / Conglomerate"] == pytest.approx(0.4)    # RELIANCE


# --------------------------------------------------------------------------- #
# New intents: routing + tools
# --------------------------------------------------------------------------- #
def test_performance_intent_sharpe(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert orch.classify_intent("What is my Sharpe ratio?")["intent"] == "performance"
    assert orch.classify_intent("What's my annualized return?")["intent"] == "performance"
    assert orch.classify_intent("Did I beat the Nifty?")["intent"] == "performance"


def test_risk_intent_volatility(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert orch.classify_intent("What is my volatility?")["intent"] == "risk"
    assert orch.classify_intent("Show my drawdown and VaR")["intent"] == "risk"


@pytest.mark.parametrize("message", [
    "Summarize my portfolio",
    "Give me an overview",
    "What are my top holdings?",
    "Show my current allocation",
])
def test_summary_intent(message, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert orch.classify_intent(message)["intent"] == "summary"


@pytest.mark.parametrize("message", [
    "What is the weight of Reliance?",
    "How much do I have in TCS?",
    "What is my allocation to Infosys?",
    "Show my RELIANCE.NS holding",
])
def test_holding_lookup_intent(message, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert orch.classify_intent(message)["intent"] == "holding_lookup"


def test_run_summary_analysis_tool():
    port = Portfolio(holdings=[Holding(ticker="RELIANCE.NS", weight=0.4),
                               Holding(ticker="TCS.NS", weight=0.35),
                               Holding(ticker="INFY.NS", weight=0.25)])
    r = tools.run_summary_analysis(port)
    assert r["num_holdings"] == 3
    assert r["concentration"]["largest_position"] == "RELIANCE.NS"
    assert "Unknown" not in r["sector_exposure"]
    json.dumps(r)


def test_run_holding_lookup_tool():
    port = Portfolio(holdings=[Holding(ticker="RELIANCE.NS", weight=0.5),
                               Holding(ticker="TCS.NS", weight=0.5)])
    r = tools.run_holding_lookup_analysis(port, "How much do I have in Reliance?")
    assert r["found"] is True
    assert r["ticker"] == "RELIANCE.NS"
    assert r["weight"] == pytest.approx(0.5)
    miss = tools.run_holding_lookup_analysis(port, "How much WIPRO do I own?")
    assert miss["found"] is False
    json.dumps(r)
