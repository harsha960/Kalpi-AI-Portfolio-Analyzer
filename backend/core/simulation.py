"""Deterministic what-if simulation.

Apply hypothetical portfolio edits (set weights, or exit a holding and move its
weight elsewhere) and return a before/after view. Pure Python — no LLM and no
metric math here; the agent calls this, then calls the metric tools separately.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..models.schemas import Holding, Portfolio

__all__ = [
    "SimulationError",
    "apply_weight_change",
    "exit_and_reallocate",
    "simulate_what_if",
]


class SimulationError(ValueError):
    """Raised when a what-if edit cannot be applied."""


# Common natural-language instrument aliases -> tickers (best-effort, optional).
_NAME_TO_TICKER = {
    "gold": "GOLDBEES.NS",
    "silver": "SILVERBEES.NS",
    "nifty": "^NSEI",
    "nifty50": "^NSEI",
    "sensex": "^BSESN",
}

_EXIT_KEYWORDS = ("exit", "sell", "remove", "drop", "out of", "reduce")
_TARGET_KEYWORDS = (" to ", " into ", " towards ", " with ", "allocate to", "move to", "buy ")
_LARGEST_PHRASES = (
    "largest position", "biggest position", "largest holding", "biggest holding",
    "top holding", "largest stock", "biggest stock", "largest allocation", "largest weight",
)
_PARTIAL_REDUCE_WORDS = ("reduce", "trim", "decrease", "cut", "lower", "shave")


# --------------------------------------------------------------------------- #
# Core deterministic edits
# --------------------------------------------------------------------------- #
def apply_weight_change(portfolio: Portfolio, changes: Mapping) -> Portfolio:
    """Set absolute target weights for the given tickers, then normalize.

    Tickers not in ``changes`` keep their current weight; a ticker set to 0 is
    removed; new tickers are added.
    """
    if not isinstance(changes, Mapping) or not changes:
        raise SimulationError("changes must be a non-empty {ticker: weight} mapping.")
    weights = dict(portfolio.weight_map())
    for ticker, new_weight in changes.items():
        t = _norm_ticker(ticker)
        if not t:
            raise SimulationError("Ticker in changes must not be empty.")
        w = float(new_weight)
        if w < 0:
            raise SimulationError(f"Weight for {t} must be non-negative, got {w}.")
        weights[t] = w
    return _weights_to_portfolio(weights)


def exit_and_reallocate(portfolio: Portfolio, exit_ticker: str, target_ticker: str) -> Portfolio:
    """Set ``exit_ticker`` weight to 0 and move its full weight to ``target_ticker``.

    If the target already exists its weight is increased; otherwise it is added
    as a new holding. The result is normalized.
    """
    weights = dict(portfolio.weight_map())
    resolved = _resolve_existing(exit_ticker, list(weights.keys()))
    if resolved is None:
        raise SimulationError(
            f"Exit ticker '{exit_ticker}' is not in the portfolio "
            f"(holdings: {list(weights.keys())})."
        )
    target = _norm_ticker(target_ticker)
    if not target:
        raise SimulationError("A non-empty target ticker is required to reallocate into.")

    moved = float(weights[resolved])
    weights[resolved] = 0.0
    weights[target] = weights.get(target, 0.0) + moved
    return _weights_to_portfolio(weights)


def simulate_what_if(
    portfolio: Portfolio,
    prices=None,
    instruction: Optional[str] = None,
    exit_ticker: Optional[str] = None,
    target_ticker: Optional[str] = None,
) -> Dict[str, object]:
    """Apply an exit-and-reallocate edit and return a before/after summary dict.

    Either pass ``exit_ticker``/``target_ticker`` explicitly, or pass a natural
    language ``instruction`` which is parsed deterministically (no LLM). If the
    edit can't be identified, returns ``changed=False`` with an explanatory note
    instead of raising.
    """
    before = portfolio.normalized()
    before_map = before.weight_map()
    notes: List[str] = []

    if (not exit_ticker or not target_ticker) and instruction:
        # Partial rebalances ("reduce TCS by 10%") aren't supported -> ask, don't guess.
        if _is_partial_move(instruction):
            p_exit, p_target, _ = _parse_instruction(instruction, before.tickers, before.weight_map())
            a = p_exit or "that holding"
            b = p_target or "the target"
            return {
                "before_portfolio": _round_map(before_map),
                "after_portfolio": _round_map(before_map),
                "changed": False,
                "note": f"Please confirm how much weight you want to move from {a} to {b}.",
                "exit_ticker": p_exit,
                "target_ticker": p_target,
            }
        p_exit, p_target, p_notes = _parse_instruction(instruction, before.tickers, before.weight_map())
        exit_ticker = exit_ticker or p_exit
        target_ticker = target_ticker or p_target
        notes.extend(p_notes)

    resolved_exit = _resolve_existing(exit_ticker, before.tickers) if exit_ticker else None

    if not resolved_exit or not target_ticker:
        if exit_ticker and not resolved_exit:
            reason = f"'{exit_ticker}' is not a current holding ({before.tickers}). "
        else:
            reason = ""
        note = (
            reason
            + "Could not determine the change to simulate. Specify which holding to "
            "exit and where to reallocate (e.g. exit_ticker and target_ticker)."
        )
        return {
            "before_portfolio": _round_map(before_map),
            "after_portfolio": _round_map(before_map),
            "changed": False,
            "note": " ".join([note] + notes).strip(),
            "exit_ticker": exit_ticker,
            "target_ticker": target_ticker,
        }

    after = exit_and_reallocate(before, resolved_exit, target_ticker)
    target_norm = _norm_ticker(target_ticker)

    price_cols = _price_columns(prices)
    if price_cols is not None and target_norm.upper() not in {c.upper() for c in price_cols}:
        notes.append(
            f"No price history available for target '{target_norm}'; "
            "its forward metrics can't be computed until data is provided."
        )

    note = f"Exited {resolved_exit} and reallocated its weight to {target_norm}."
    if notes:
        note = note + " " + " ".join(notes)

    return {
        "before_portfolio": _round_map(before_map),
        "after_portfolio": _round_map(after.weight_map()),
        "changed": True,
        "note": note,
        "exit_ticker": resolved_exit,
        "target_ticker": target_norm,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _weights_to_portfolio(weights: Mapping) -> Portfolio:
    holdings = [Holding(ticker=t, weight=w) for t, w in weights.items() if float(w) > 0.0]
    if not holdings:
        raise SimulationError("Resulting portfolio has no positive-weight holdings.")
    return Portfolio(holdings=holdings).normalized()


def _norm_ticker(ticker) -> str:
    if ticker is None:
        return ""
    return str(ticker).strip().upper()


def _resolve_existing(ticker, existing) -> Optional[str]:
    """Match a (possibly informal) ticker to an existing holding by symbol/base."""
    if ticker is None:
        return None
    t = str(ticker).strip().upper()
    if not t:
        return None
    by_symbol = {e.upper(): e for e in existing}
    if t in by_symbol:
        return by_symbol[t]
    by_base = {e.split(".")[0].upper(): e for e in existing}
    if t.split(".")[0] in by_base:
        return by_base[t.split(".")[0]]
    return None


def _price_columns(prices) -> Optional[List[str]]:
    if prices is None:
        return None
    if hasattr(prices, "prices") and isinstance(getattr(prices, "prices"), pd.DataFrame):
        prices = prices.prices
    if isinstance(prices, pd.DataFrame):
        return [str(c) for c in prices.columns]
    return None


def _round_map(weight_map: Mapping) -> Dict[str, float]:
    return {str(k): round(float(v), 6) for k, v in weight_map.items()}


def _is_partial_move(instruction: str) -> bool:
    """True for partial-rebalance phrasing like 'reduce TCS by 10%' (not supported)."""
    text = str(instruction).lower()
    has_reduce = any(word in text for word in _PARTIAL_REDUCE_WORDS)
    has_amount = ("%" in text) or bool(re.search(r"\bby\s+\d", text)) or bool(re.search(r"\d+\s*%", text))
    return has_reduce and has_amount


def _parse_instruction(instruction: str, tickers: List[str], weight_map: Optional[Mapping] = None) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Best-effort, deterministic extraction of (exit, target) tickers from text."""
    text = " " + str(instruction).lower() + " "
    notes: List[str] = []

    mentions: Dict[str, int] = {}
    for ticker in tickers:
        for key in (ticker.lower(), ticker.split(".")[0].lower()):
            pos = text.find(key)
            if pos != -1:
                mentions.setdefault(ticker, pos)

    # Exit: first holding mentioned after an exit keyword, else earliest mention.
    exit_ticker = None
    for kw in _EXIT_KEYWORDS:
        kpos = text.find(kw)
        if kpos != -1:
            after = {t: p for t, p in mentions.items() if p >= kpos}
            if after:
                exit_ticker = min(after, key=after.get)
                break
    if exit_ticker is None and mentions:
        exit_ticker = min(mentions, key=mentions.get)
    # "exit my largest position" -> resolve to the max-weight holding.
    if exit_ticker is None and weight_map and any(p in text for p in _LARGEST_PHRASES):
        exit_ticker = max(weight_map, key=weight_map.get)

    # Target: token following a reallocation keyword.
    target = None
    for kw in _TARGET_KEYWORDS:
        kpos = text.find(kw)
        if kpos != -1:
            parts = text[kpos + len(kw):].strip().split()
            if parts:
                word = parts[0].strip(" .,:;?!")
                if word:
                    target = _NAME_TO_TICKER.get(word) or _resolve_existing(word, tickers) or word.upper()
                    break

    if target is not None and exit_ticker is not None and target == exit_ticker:
        notes.append("Parsed target matches the exited holding; please clarify the target.")
    return exit_ticker, target, notes
