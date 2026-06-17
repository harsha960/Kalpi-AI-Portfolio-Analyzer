"""FastAPI application for the Kalpi AI Portfolio Analyzer.

Thin HTTP layer: validate input, classify intent (agent), fetch market data,
call the deterministic tools, and format a short human-readable answer from the
tool numbers. The LLM only ever picks an intent label — every number returned to
the user comes from deterministic Python (the "Golden Rule"). No DB, no auth.
"""
from __future__ import annotations

from typing import Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agent import tools
from .agent.orchestrator import classify_intent
from .config import settings
from .core.data import MarketDataError, fetch_benchmark_history, get_portfolio_prices
from .core.diversification import DiversificationError
from .core.performance import PerformanceError
from .core.portfolio import PortfolioError, parse_portfolio_text
from .core.risk import RiskError
from .core.simulation import SimulationError
from .models.schemas import ChatRequest, ParsePortfolioRequest, Portfolio

app = FastAPI(title=settings.SERVICE_NAME, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Deterministic core errors -> 400 (client can fix the input).
_CORE_ERRORS = (PortfolioError, PerformanceError, RiskError, DiversificationError, SimulationError)

_SUGGESTIONS = [
    "How has my portfolio performed versus the Nifty 50?",
    "What is my portfolio's risk - volatility, drawdown and VaR?",
    "Am I over-concentrated in any sector?",
    "Show me the correlation matrix of my holdings.",
    "What if I exit my largest position and move it to gold?",
]
_GENERAL_ANSWER = (
    "I could not confidently classify that question. Try asking about performance, "
    "risk, sector concentration, correlations, or what-if simulations."
)
_MISSING_PORTFOLIO_MSG = (
    "I need your portfolio before I can answer this. Please upload a CSV or paste "
    "holdings like:\n\nTicker,Weight\nRELIANCE.NS,25\nTCS.NS,20\nINFY.NS,15"
)


def _missing_portfolio_response(intent: str, classification: dict) -> dict:
    return {
        "intent": intent,
        "classification": classification,
        "answer": _MISSING_PORTFOLIO_MSG,
        "metrics": {},
        "chart_data": {"type": "need_portfolio"},
        "warnings": [],
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "service": settings.SERVICE_NAME}


@app.post("/parse_portfolio")
def parse_portfolio_endpoint(req: ParsePortfolioRequest):
    try:
        portfolio = parse_portfolio_text(req.text)
    except PortfolioError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "portfolio": [{"ticker": h.ticker, "weight": h.weight} for h in portfolio.holdings],
        "total_weight": round(portfolio.total_weight(), 6),
        "message": "Portfolio parsed successfully",
    }


@app.post("/chat")
def chat(req: ChatRequest):
    classification = classify_intent(req.message)
    intent = classification["intent"]
    warnings: list = []

    # 'general' needs no portfolio and no market data.
    if intent == "general":
        return {
            "intent": "general",
            "classification": classification,
            "answer": _GENERAL_ANSWER,
            "metrics": {},
            "chart_data": {"type": "suggestions", "suggestions": _SUGGESTIONS},
            "warnings": warnings,
        }

    # Summary and single-holding lookups need a portfolio but NO market data.
    if intent in ("summary", "holding_lookup"):
        if not req.portfolio:
            return _missing_portfolio_response(intent, classification)
        try:
            portfolio = Portfolio(holdings=req.portfolio).normalized()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid portfolio: {exc}")
        if intent == "summary":
            result = tools.run_summary_analysis(portfolio)
        else:
            result = tools.run_holding_lookup_analysis(portfolio, req.message)
        answer, metrics, chart_data = _present(intent, result)
        return {
            "intent": intent,
            "classification": classification,
            "answer": answer,
            "metrics": metrics,
            "chart_data": chart_data,
            "warnings": warnings,
        }

    # Remaining analytical intents need a portfolio AND market data.
    if not req.portfolio:
        return _missing_portfolio_response(intent, classification)
    try:
        portfolio = Portfolio(holdings=req.portfolio).normalized()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid portfolio: {exc}")

    # Fetch market data (graceful failures).
    period = req.period or settings.PRICE_HISTORY_PERIOD
    try:
        price_history = get_portfolio_prices(portfolio, period=period)
    except MarketDataError as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch market data: {exc}")
    except Exception as exc:  # network/library failure
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching market data: {exc}")

    prices = price_history.prices
    if price_history.missing:
        warnings.append(
            f"No price data for: {', '.join(price_history.missing)}. Excluded from the analysis."
        )

    # Dispatch to the matching deterministic tool.
    try:
        if intent == "performance":
            benchmark_prices = None
            try:
                benchmark_prices = fetch_benchmark_history(settings.BENCHMARK_TICKER, period=period).prices
            except Exception as bexc:
                warnings.append(f"Benchmark ({settings.BENCHMARK_TICKER}) unavailable: {bexc}")
            result = tools.run_performance_analysis(portfolio, prices, benchmark_prices=benchmark_prices)
        elif intent == "risk":
            result = tools.run_risk_analysis(portfolio, prices)
        elif intent == "diversification":
            result = tools.run_diversification_analysis(portfolio, prices)
        elif intent == "correlation":
            result = tools.run_correlation_analysis(portfolio, prices)
        elif intent == "what_if":
            result = tools.run_what_if_analysis(portfolio, prices, req.message)
        else:
            result = {}
    except _CORE_ERRORS as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    answer, metrics, chart_data = _present(intent, result)
    return {
        "intent": intent,
        "classification": classification,
        "answer": answer,
        "metrics": metrics,
        "chart_data": chart_data,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Deterministic answer/chart formatting (numbers come only from tool outputs)
# --------------------------------------------------------------------------- #
def _pct(value, dp: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _most_correlated(matrix: dict):
    cols = list(matrix.keys())
    best = None
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = matrix.get(a, {}).get(b)
            if v is None:
                continue
            if best is None or v > best[2]:
                best = (a, b, float(v))
    return best


def _present(intent: str, result: dict) -> Tuple[str, dict, dict]:
    if intent == "performance":
        m = result["metrics"]
        answer = (
            f"Over {m['observations']} trading days, the portfolio's total return is "
            f"{_pct(m['total_return'])} (annualized {_pct(m['annualized_return'])}), with annualized "
            f"volatility {_pct(m['annualized_volatility'])} and a Sharpe ratio of {m['sharpe_ratio']:.2f}."
        )
        metrics = dict(m)
        chart = {"type": "performance", "cumulative_return": result.get("cumulative_return", {})}
        if "benchmark" in result:
            b = result["benchmark"]
            pf, bm = b["portfolio"], b["benchmark"]
            verb = "outperformed" if b["outperformed"] else "underperformed"
            answer += (
                f" Versus {bm['name']}, it {verb} on an annualized basis "
                f"({_pct(pf['annualized_return'])} vs {_pct(bm['annualized_return'])})."
            )
            metrics["benchmark"] = b
            chart["benchmark"] = b
        return answer, metrics, chart

    if intent == "risk":
        m = result["metrics"]
        answer = (
            f"Annualized volatility is {_pct(m['annualized_volatility'])}, with a maximum drawdown of "
            f"{_pct(m['max_drawdown'])}. The 1-day 95% historical VaR is {_pct(m['historical_var_95'], 2)} "
            f"(CVaR {_pct(m['historical_cvar_95'], 2)}); downside deviation is {_pct(m['downside_deviation'])}."
        )
        chart = {"type": "risk", "drawdown": result.get("drawdown", {}), "metrics": m}
        return answer, dict(m), chart

    if intent == "diversification":
        c = result["concentration"]
        sectors = result["sector_exposure"]
        wavg = result["weighted_avg_correlation"]
        answer = (
            f"Your largest position is {c['largest_position']} at {_pct(c['max_weight'])}, and the top 3 "
            f"holdings make up {_pct(c['top_3_weight'])} (HHI {c['hhi']:.2f}). The weighted-average pairwise "
            f"correlation is {wavg:.2f}."
        )
        if sectors:
            top = next(iter(sectors))
            answer += f" Largest sector exposure: {top} at {_pct(sectors[top])}."
        metrics = {"concentration": c, "weighted_avg_correlation": wavg, "sector_exposure": sectors}
        chart = {
            "type": "diversification",
            "sector_exposure": sectors,
            "correlation_matrix": result["correlation_matrix"],
            "concentration": c,
        }
        return answer, metrics, chart

    if intent == "correlation":
        matrix = result["correlation_matrix"]
        wavg = result["weighted_avg_correlation"]
        answer = (
            f"The weighted-average pairwise correlation across your holdings is {wavg:.2f}. "
            "See the correlation matrix for pair-level detail."
        )
        pair = _most_correlated(matrix)
        if pair:
            answer += f" Most correlated pair: {pair[0]} & {pair[1]} ({pair[2]:.2f})."
        return answer, {"weighted_avg_correlation": wavg}, {"type": "correlation", "correlation_matrix": matrix}

    if intent == "what_if":
        sim_r = result["simulation"]
        answer = sim_r.get("note", "Simulation complete.")
        metrics: dict = {}
        if "before_metrics" in result:
            metrics["before"] = result["before_metrics"]
        if "after_metrics" in result:
            metrics["after"] = result["after_metrics"]
        if "before_metrics" in result and "after_metrics" in result:
            bm, am = result["before_metrics"], result["after_metrics"]
            answer += (
                f" Sharpe {bm['sharpe_ratio']:.2f} -> {am['sharpe_ratio']:.2f}; "
                f"volatility {_pct(bm['annualized_volatility'])} -> {_pct(am['annualized_volatility'])}."
            )
        chart = {
            "type": "what_if",
            "before_portfolio": sim_r.get("before_portfolio", {}),
            "after_portfolio": sim_r.get("after_portfolio", {}),
        }
        return answer, metrics, chart

    if intent == "summary":
        c = result.get("concentration", {})
        sectors = result.get("sector_exposure", {})
        n = result.get("num_holdings", len(result.get("allocation", {})))
        answer = (
            f"Your portfolio has {n} holdings. The largest position is "
            f"{c.get('largest_position', '-')} at {_pct(c.get('max_weight'))}. The top 3 "
            f"holdings make up {_pct(c.get('top_3_weight'))}."
        )
        if sectors:
            top = next(iter(sectors))
            answer += f" Largest sector exposure is {top} at {_pct(sectors[top])}."
        metrics = {"num_holdings": n, "concentration": c, "sector_exposure": sectors}
        chart = {"type": "summary", "allocation": result.get("allocation", {}),
                 "sector_exposure": sectors, "concentration": c}
        return answer, metrics, chart

    if intent == "holding_lookup":
        if result.get("found"):
            answer = f"{result['ticker']} is {_pct(result['weight'])} of your portfolio."
        else:
            available = ", ".join(result.get("available", [])) or "none"
            answer = ("I couldn't find that holding in your portfolio. "
                      f"Your holdings are: {available}.")
        metrics = {"ticker": result.get("ticker"), "weight": result.get("weight"),
                   "found": result.get("found")}
        chart = {"type": "holding_lookup", "allocation": result.get("allocation", {}),
                 "highlight": result.get("ticker")}
        return answer, metrics, chart

    return "Here is the analysis.", result, {"type": intent}
