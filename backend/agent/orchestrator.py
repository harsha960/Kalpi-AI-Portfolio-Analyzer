"""Agent intent router: classify a user message into a supported intent.

THE GOLDEN RULE: the LLM (when available) is used ONLY to pick an intent label.
It never computes or guesses any portfolio metric. With no API key — or if the
LLM call fails — a deterministic keyword classifier is used, so the app always
works offline.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

__all__ = ["SUPPORTED_INTENTS", "classify_intent"]

SUPPORTED_INTENTS: List[str] = [
    "performance", "risk", "diversification", "correlation",
    "what_if", "summary", "holding_lookup", "general",
]

# Rule-based keyword sets (substring match, case-insensitive).
_RULES: Dict[str, List[str]] = {
    "performance": ["return", "performance", "benchmark", "nifty", "growth", "profit",
                    "sharpe", "annualized", "annualised", "beat", "cagr", "alpha"],
    "risk": ["risk", "volatility", "volatile", "drawdown", "var", "cvar", "loss", "downside"],
    "diversification": ["diversify", "diversification", "sector", "exposure", "concentration", "allocation"],
    "correlation": ["correlation", "correlated", "matrix", "overlap"],
    "what_if": ["what if", "simulate", "exit", "replace", "rebalance", "allocate", "move", "reallocate"],
    "summary": ["summarize", "summarise", "summary", "overview", "top holding", "top holdings",
                "current allocation", "portfolio breakdown", "give me an overview"],
    "holding_lookup": ["weight of", "how much do i have", "how much do i hold", "allocation to",
                       "how much in", "weight in", "position in", "how much of",
                       "do i hold", "do i own", "holding"],
}

# Tie-break priority, most specific first ("general" is the fallback).
_PRIORITY: List[str] = [
    "what_if", "summary", "holding_lookup", "correlation", "risk", "performance", "diversification",
]


def classify_intent(message: str, use_llm: bool = True) -> Dict[str, object]:
    """Return ``{"intent", "confidence", "method"}`` for a user message."""
    text = (message or "").strip()
    if not text:
        return {"intent": "general", "confidence": 0.0, "method": "rules"}

    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            intent = _classify_with_llm(text)
            if intent in SUPPORTED_INTENTS:
                return {"intent": intent, "confidence": 0.9, "method": "llm"}
        except Exception:
            pass  # any failure -> deterministic fallback

    intent, confidence = _classify_with_rules(text)
    return {"intent": intent, "confidence": confidence, "method": "rules"}


def _classify_with_rules(message: str) -> Tuple[str, float]:
    text = message.lower()
    scores = {
        intent: sum(1 for kw in keywords if kw in text)
        for intent, keywords in _RULES.items()
    }
    best_score = max(scores.values())
    if best_score == 0:
        return "general", 0.3
    best_intent = next(i for i in _PRIORITY if scores.get(i, 0) == best_score)
    confidence = min(0.95, 0.5 + 0.15 * best_score)
    return best_intent, round(confidence, 3)


def _classify_with_llm(message: str) -> str:
    """Use OpenAI ONLY to label intent; never to compute metrics. Lazy-imports openai."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system = (
        "You are an intent classifier for a portfolio analytics app. Classify the "
        "user's message into exactly one of these labels: performance, risk, "
        "diversification, correlation, what_if, summary, holding_lookup, general. "
        "Use 'summary' for overview/top-holdings/allocation requests, 'holding_lookup' "
        "for questions about a single holding's weight. Reply with only the single "
        "label word, nothing else."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=4,
    )
    raw = (response.choices[0].message.content or "").strip().lower()
    return _coerce_intent(raw)


def _coerce_intent(text: str) -> str:
    text = text.strip().lower()
    if text in SUPPORTED_INTENTS:
        return text
    for intent in SUPPORTED_INTENTS:
        if intent in text:
            return intent
    return "general"
