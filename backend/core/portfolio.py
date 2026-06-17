"""Deterministic portfolio ingestion: parse, validate, normalize.

Turns a CSV upload or pasted text into a validated, weight-normalized
``Portfolio``. Pure and offline — no market data and no metrics here.
"""
from __future__ import annotations

import io
import re
from typing import IO, Iterable, List, Sequence, Tuple, Union

import pandas as pd

from ..models.schemas import Holding, Portfolio

__all__ = [
    "PortfolioError",
    "parse_portfolio_csv",
    "parse_portfolio_text",
    "dataframe_to_portfolio",
]


class PortfolioError(ValueError):
    """Raised when input cannot be parsed into a valid portfolio."""


# Column-name aliases, matched case-insensitively (ignoring spaces/punctuation).
_TICKER_ALIASES = ("ticker", "tickers", "symbol", "symbols", "stock", "scrip", "security")
_WEIGHT_ALIASES = (
    "weight", "weights", "weightage", "allocation", "alloc",
    "quantity", "qty", "units", "shares",
)

CsvSource = Union[bytes, str, IO]


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def parse_portfolio_csv(source: CsvSource) -> Portfolio:
    """Parse an uploaded CSV (bytes / str / file-like) into a normalized Portfolio."""
    buffer = _as_text_buffer(source)
    try:
        df = pd.read_csv(buffer)
    except Exception as exc:  # malformed CSV
        raise PortfolioError(f"Could not read CSV: {exc}") from exc
    return dataframe_to_portfolio(df)


def parse_portfolio_text(text: str) -> Portfolio:
    """Parse pasted text (header optional) into a normalized Portfolio."""
    if text is None or not str(text).strip():
        raise PortfolioError("No portfolio text provided.")
    df = _text_to_dataframe(str(text))
    return dataframe_to_portfolio(df)


def dataframe_to_portfolio(df: pd.DataFrame) -> Portfolio:
    """Validate a frame with Ticker/Weight columns and return a normalized Portfolio."""
    if df is None or df.empty:
        raise PortfolioError("Portfolio is empty.")

    ticker_col, weight_col = _resolve_columns(df)
    raw = pd.DataFrame(
        {
            "Ticker": df[ticker_col].map(_clean_ticker),
            "Weight": _coerce_weights(df[weight_col]),
        }
    ).reset_index(drop=True)

    # Drop fully blank lines (e.g. trailing newlines in pasted text).
    raw = raw[~(raw["Ticker"].eq("") & raw["Weight"].isna())]
    if raw.empty:
        raise PortfolioError("No holdings found.")

    # Validate tickers are not empty.
    empty_mask = raw["Ticker"].eq("")
    if empty_mask.any():
        rows = [int(i) + 1 for i in raw.index[empty_mask]]
        raise PortfolioError(f"Empty ticker(s) in row(s): {rows}.")

    # Validate weights are positive numbers.
    nan_mask = raw["Weight"].isna()
    if nan_mask.any():
        bad = raw.loc[nan_mask, "Ticker"].tolist()
        raise PortfolioError(f"Non-numeric or missing weight for: {bad}.")
    nonpos_mask = raw["Weight"] <= 0
    if nonpos_mask.any():
        bad = raw.loc[nonpos_mask, "Ticker"].tolist()
        raise PortfolioError(f"Weights must be positive numbers; check: {bad}.")

    # Aggregate duplicate tickers (e.g. two lots of the same stock).
    grouped = raw.groupby("Ticker", as_index=False, sort=False)["Weight"].sum()

    holdings = [
        Holding(ticker=t, weight=float(w))
        for t, w in zip(grouped["Ticker"], grouped["Weight"])
    ]
    # Holding/Portfolio re-validate, then we normalize so weights sum to 1.
    return Portfolio(holdings=holdings).normalized()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_text_buffer(source: CsvSource) -> IO:
    """Coerce bytes/str/file-like into a text buffer (handles Excel BOM)."""
    if isinstance(source, bytes):
        return io.StringIO(source.decode("utf-8-sig", errors="replace"))
    if isinstance(source, str):
        return io.StringIO(source)
    data = source.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig", errors="replace")
    return io.StringIO(data)


def _resolve_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Find the ticker and weight columns; fall back to first-two positional."""
    lookup = {_norm(c): c for c in df.columns}
    ticker_col = _first_match(lookup, _TICKER_ALIASES)
    weight_col = _first_match(lookup, _WEIGHT_ALIASES)

    if ticker_col is None or weight_col is None:
        if df.shape[1] >= 2:  # unlabeled: assume [Ticker, Weight] order
            return df.columns[0], df.columns[1]
        missing = []
        if ticker_col is None:
            missing.append("Ticker")
        if weight_col is None:
            missing.append("Weight")
        raise PortfolioError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Found columns: {list(df.columns)}."
        )
    return ticker_col, weight_col


def _first_match(lookup: dict, aliases: Sequence[str]):
    for alias in aliases:
        key = _norm(alias)
        if key in lookup:
            return lookup[key]
    return None


def _norm(col) -> str:
    """Lowercase and strip non-alphanumerics, so 'Weight (%)' -> 'weight'."""
    return re.sub(r"[^a-z0-9]", "", str(col).strip().lower())


def _clean_ticker(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().upper()


def _coerce_weights(series: pd.Series) -> pd.Series:
    """Numeric weights, tolerating strings like '25%' or '1,000'."""
    if series.dtype == object:
        series = (
            series.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    return pd.to_numeric(series, errors="coerce")


def _text_to_dataframe(text: str) -> pd.DataFrame:
    """Parse pasted text into a DataFrame, with or without a header row."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise PortfolioError("No portfolio text provided.")

    rows = [parts for ln in lines if (parts := _split_row(ln))]
    has_header = _looks_like_header(lines[0])

    if has_header:
        header = [str(h) for h in rows[0]]
        width = len(header)
        body = [(r + [""] * width)[:width] for r in rows[1:]]
        return pd.DataFrame(body, columns=header)

    # Headerless: take the first two positional fields as Ticker, Weight.
    body = [(r + ["", ""])[:2] for r in rows]
    return pd.DataFrame(body, columns=["Ticker", "Weight"])


def _split_row(line: str) -> List[str]:
    """Split a row on comma, then tab, then whitespace."""
    line = line.strip()
    if not line:
        return []
    if "," in line:
        parts = line.split(",")
    elif "\t" in line:
        parts = line.split("\t")
    else:
        parts = line.split()
    return [p.strip() for p in parts]


def _looks_like_header(line: str) -> bool:
    tokens = {_norm(p) for p in _split_row(line)}
    alias_tokens = {_norm(a) for a in (_TICKER_ALIASES + _WEIGHT_ALIASES)}
    return bool(tokens & alias_tokens)
