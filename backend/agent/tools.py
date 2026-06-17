"""Deterministic tool functions the agent / FastAPI layer will call.

Each tool composes the pure functions in backend.core and returns a
JSON-serializable dict. No LLM and no metric math live here — these are thin
orchestration wrappers, so the agent never computes anything itself.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..core import diversification as dv
from ..core import performance as perf
from ..core import risk as rk
from ..core import simulation as sim

__all__ = [
    "DEFAULT_SECTOR_MAP",
    "NAME_ALIASES",
    "run_performance_analysis",
    "run_risk_analysis",
    "run_diversification_analysis",
    "run_correlation_analysis",
    "run_what_if_analysis",
    "run_summary_analysis",
    "run_holding_lookup_analysis",
]

# Offline sector map for common Indian (NSE) tickers, so the demo shows real
# sector exposure without a (slow, optional) yfinance lookup. Override per call
# by passing a custom sector_map to run_diversification_analysis / run_summary_analysis.
DEFAULT_SECTOR_MAP = {
    "RELIANCE.NS": "Energy / Conglomerate",
    "TCS.NS": "Information Technology",
    "INFY.NS": "Information Technology",
    "WIPRO.NS": "Information Technology",
    "HCLTECH.NS": "Information Technology",
    "HDFCBANK.NS": "Financial Services",
    "ICICIBANK.NS": "Financial Services",
    "SBIN.NS": "Financial Services",
    "KOTAKBANK.NS": "Financial Services",
    "AXISBANK.NS": "Financial Services",
    "BAJFINANCE.NS": "Financial Services",
    "ITC.NS": "FMCG",
    "HINDUNILVR.NS": "FMCG",
    "NESTLEIND.NS": "FMCG",
    "SUNPHARMA.NS": "Healthcare",
    "DRREDDY.NS": "Healthcare",
    "LT.NS": "Industrials",
    "BHARTIARTL.NS": "Telecom",
    "MARUTI.NS": "Automobile",
    "TATAMOTORS.NS": "Automobile",
    "ASIANPAINT.NS": "Consumer Durables",
    "GOLDBEES.NS": "Gold / Commodity",
    "SILVERBEES.NS": "Gold / Commodity",
}

# Common name -> NSE ticker aliases for single-holding lookups.
NAME_ALIASES = {
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfcbank": "HDFCBANK.NS",
    "hdfc": "HDFCBANK.NS",
    "itc": "ITC.NS",
    "gold": "GOLDBEES.NS",
    "goldbees": "GOLDBEES.NS",
    "sun pharma": "SUNPHARMA.NS",
    "sunpharma": "SUNPHARMA.NS",
    "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "airtel": "BHARTIARTL.NS",
}


def run_performance_analysis(portfolio, prices, benchmark_prices=None) -> dict:
    port_returns = perf.calculate_portfolio_returns(prices, portfolio)
    out = {
        "intent": "performance",
        "metrics": perf.performance_summary(prices, portfolio),
        "cumulative_return": _series_to_jsonable(perf.calculate_cumulative_returns(port_returns)),
    }
    if benchmark_prices is not None:
        try:
            out["benchmark"] = perf.benchmark_comparison(prices, benchmark_prices, portfolio)
        except Exception as exc:
            out["benchmark_error"] = str(exc)
    return out


def run_risk_analysis(portfolio, prices) -> dict:
    port_returns = perf.calculate_portfolio_returns(prices, portfolio)
    return {
        "intent": "risk",
        "metrics": rk.risk_summary(port_returns),
        "drawdown": _series_to_jsonable(rk.drawdown_series(port_returns)),
    }


def run_diversification_analysis(portfolio, prices, sector_map=None) -> dict:
    effective_map = sector_map if sector_map is not None else DEFAULT_SECTOR_MAP
    summary = dv.diversification_summary(prices, portfolio, portfolio=portfolio, sector_map=effective_map)
    return {"intent": "diversification", **summary}


def run_correlation_analysis(portfolio, prices) -> dict:
    corr = dv.correlation_matrix(prices)
    return {
        "intent": "correlation",
        "correlation_matrix": _matrix_to_jsonable(corr),
        "weighted_avg_correlation": float(dv.weighted_correlation_score(prices, portfolio)),
    }


def run_what_if_analysis(portfolio, prices, message) -> dict:
    sim_result = sim.simulate_what_if(portfolio, prices, instruction=message)
    out = {"intent": "what_if", "simulation": sim_result}

    try:
        out["before_metrics"] = perf.performance_summary(prices, portfolio)
    except Exception as exc:
        out["before_metrics_error"] = str(exc)

    if sim_result.get("changed"):
        frame = _price_frame(prices)
        cols = {str(c) for c in frame.columns} if frame is not None else set()
        usable = {t: w for t, w in sim_result["after_portfolio"].items() if t in cols and w > 0}
        if usable and frame is not None:
            try:
                out["after_metrics"] = perf.performance_summary(frame[list(usable.keys())], usable)
            except Exception as exc:
                out["after_metrics_error"] = str(exc)
        else:
            out["after_metrics_note"] = (
                "No price data for the resulting holdings; after-metrics unavailable."
            )
    return out


def run_summary_analysis(portfolio, sector_map=None) -> dict:
    """Deterministic portfolio overview — no market data needed."""
    effective_map = sector_map if sector_map is not None else DEFAULT_SECTOR_MAP
    concentration = dv.concentration_metrics(portfolio)
    sectors = dv.sector_exposure(portfolio, sector_map=effective_map)
    return {
        "intent": "summary",
        "num_holdings": concentration["num_holdings"],
        "concentration": concentration,
        "sector_exposure": sectors,
        "allocation": concentration["weights"],
    }


def run_holding_lookup_analysis(portfolio, message) -> dict:
    """Look up the weight of a single holding by ticker or common name — no market data."""
    weight_map = portfolio.normalized().weight_map()
    resolved = _resolve_holding(message, list(weight_map.keys()))
    if resolved is None or resolved not in weight_map:
        return {
            "intent": "holding_lookup",
            "found": False,
            "ticker": resolved,
            "weight": None,
            "available": list(weight_map.keys()),
            "allocation": weight_map,
        }
    return {
        "intent": "holding_lookup",
        "found": True,
        "ticker": resolved,
        "weight": float(weight_map[resolved]),
        "allocation": weight_map,
    }


# --------------------------- helpers ---------------------------
def _resolve_holding(message, tickers):
    """Deterministically match a message to one of the portfolio's tickers."""
    text = " " + str(message).lower() + " "
    candidates = []
    for t in tickers:
        for key in (t.lower(), t.split(".")[0].lower()):
            i = text.find(key)
            if i != -1:
                candidates.append((i, t))
                break
    for alias, mapped in NAME_ALIASES.items():
        i = text.find(alias)
        if i != -1 and mapped in tickers:
            candidates.append((i, mapped))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])  # earliest mention wins
    return candidates[0][1]


def _series_to_jsonable(series: pd.Series) -> dict:
    return {str(pd.Timestamp(idx).date()): float(val) for idx, val in series.items()}


def _matrix_to_jsonable(corr: pd.DataFrame) -> dict:
    return {
        str(col): {str(idx): (None if pd.isna(v) else float(v)) for idx, v in corr[col].items()}
        for col in corr.columns
    }


def _price_frame(prices) -> Optional[pd.DataFrame]:
    if prices is None:
        return None
    if hasattr(prices, "prices") and isinstance(getattr(prices, "prices"), pd.DataFrame):
        return prices.prices
    if isinstance(prices, pd.DataFrame):
        return prices
    return None
