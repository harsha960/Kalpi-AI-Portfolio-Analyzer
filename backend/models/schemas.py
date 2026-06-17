"""Pydantic models for the core data layer and the API request bodies.

Portfolio data structures plus the request shapes for the FastAPI routes. No
financial math and no market data live here (the project's "Golden Rule").
"""
from __future__ import annotations

import math
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["Holding", "Portfolio", "ParsePortfolioRequest", "ChatRequest"]


class Holding(BaseModel):
    """A single position: a ticker and its (pre-normalization) weight."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str = Field(..., description="Exchange ticker, e.g. 'RELIANCE.NS'.")
    weight: float = Field(..., description="Positive weight; normalized later so weights sum to 1.")

    @field_validator("ticker")
    @classmethod
    def _ticker_not_empty(cls, value: str) -> str:
        cleaned = (value or "").strip().upper()
        if not cleaned:
            raise ValueError("Ticker must not be empty.")
        return cleaned

    @field_validator("weight")
    @classmethod
    def _weight_positive(cls, value: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Weight must be a number, got {value!r}.")
        if not math.isfinite(number):
            raise ValueError(f"Weight must be a finite number, got {value!r}.")
        if number <= 0:
            raise ValueError(f"Weight must be positive, got {number}.")
        return number


class Portfolio(BaseModel):
    """A validated set of holdings. Use ``normalized()`` to make weights sum to 1.0."""

    holdings: List[Holding] = Field(..., min_length=1)

    @field_validator("holdings")
    @classmethod
    def _at_least_one(cls, value: List[Holding]) -> List[Holding]:
        if not value:
            raise ValueError("Portfolio must contain at least one holding.")
        return value

    @property
    def tickers(self) -> List[str]:
        return [h.ticker for h in self.holdings]

    @property
    def weights(self) -> List[float]:
        return [h.weight for h in self.holdings]

    def total_weight(self) -> float:
        return float(sum(self.weights))

    def weight_map(self) -> Dict[str, float]:
        return {h.ticker: h.weight for h in self.holdings}

    def is_normalized(self, tolerance: float = 1e-6) -> bool:
        return abs(self.total_weight() - 1.0) <= tolerance

    def normalized(self) -> "Portfolio":
        total = self.total_weight()
        if total <= 0:
            raise ValueError("Total weight must be positive to normalize.")
        return Portfolio(
            holdings=[Holding(ticker=h.ticker, weight=h.weight / total) for h in self.holdings]
        )


# --------------------------------------------------------------------------- #
# API request bodies
# --------------------------------------------------------------------------- #
class ParsePortfolioRequest(BaseModel):
    """Body for POST /parse_portfolio."""

    text: str = Field(..., description="CSV-like text with Ticker and Weight columns.")


class ChatRequest(BaseModel):
    """Body for POST /chat."""

    message: str = Field(..., description="The user's natural-language question.")
    portfolio: List[Holding] = Field(default_factory=list, description="Current holdings.")
    period: str = Field(default="1y", description="Price-history window (e.g. 1y, 2y, 6mo).")
