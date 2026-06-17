"""FastAPI route tests. Offline only — no live Yahoo Finance."""
import pytest
from fastapi.testclient import TestClient

import backend.core.data as data
from backend.main import app

client = TestClient(app)


def _boom(*_args, **_kwargs):
    raise AssertionError("market data should not be fetched on this path")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "Kalpi AI Portfolio Analyzer"}


def test_parse_portfolio():
    r = client.post("/parse_portfolio", json={"text": "Ticker,Weight\nRELIANCE.NS,25\nTCS.NS,20"})
    assert r.status_code == 200
    body = r.json()
    assert body["message"] == "Portfolio parsed successfully"
    assert body["total_weight"] == pytest.approx(1.0)
    weights = {h["ticker"]: h["weight"] for h in body["portfolio"]}
    assert weights["RELIANCE.NS"] == pytest.approx(25 / 45)
    assert weights["TCS.NS"] == pytest.approx(20 / 45)


def test_parse_portfolio_invalid():
    # Negative weight must be rejected by the deterministic parser -> 400.
    r = client.post("/parse_portfolio", json={"text": "Ticker,Weight\nRELIANCE.NS,-5\nTCS.NS,10"})
    assert r.status_code == 400


def test_chat_general_no_yfinance(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(data, "fetch_price_history", _boom)  # guard: must not be called
    r = client.post("/chat", json={"message": "Hello, what can you do?", "portfolio": []})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "general"
    assert body["answer"]
    assert body["chart_data"]["type"] == "suggestions"
    assert isinstance(body["chart_data"]["suggestions"], list) and body["chart_data"]["suggestions"]


def test_missing_portfolio_chat(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(data, "fetch_price_history", _boom)  # no fetch before a portfolio exists
    r = client.post("/chat", json={"message": "What is my portfolio risk?", "portfolio": []})
    assert r.status_code == 200
    body = r.json()
    assert "i need your portfolio" in body["answer"].lower()
    assert body["chart_data"]["type"] == "need_portfolio"


def test_fallback_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/chat", json={"message": "qwerty zzz floof", "portfolio": []})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "general"
    assert "could not confidently classify" in body["answer"].lower()


_PF = [
    {"ticker": "RELIANCE.NS", "weight": 0.5},
    {"ticker": "TCS.NS", "weight": 0.3},
    {"ticker": "INFY.NS", "weight": 0.2},
]


def test_chat_summary_with_portfolio(monkeypatch):
    # Summary needs a portfolio but NO market data -> fetch must not be called.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(data, "fetch_price_history", _boom)
    r = client.post("/chat", json={"message": "Summarize my portfolio", "portfolio": _PF})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "summary"
    assert "holdings" in body["answer"].lower()
    assert body["chart_data"]["type"] == "summary"


def test_chat_holding_lookup_with_portfolio(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(data, "fetch_price_history", _boom)
    r = client.post("/chat", json={"message": "What is the weight of Reliance?", "portfolio": _PF})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "holding_lookup"
    assert "RELIANCE.NS is" in body["answer"]
    assert body["chart_data"]["type"] == "holding_lookup"
