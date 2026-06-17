"""Centralized settings loaded from environment variables (.env optional).

Kept deliberately small — no database, no auth. Values fall back to sensible
defaults so the app runs out of the box (rule-based agent, Nifty 50 benchmark).
"""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv not installed -> rely on real env vars
    pass


def _split_csv(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]


class Settings:
    SERVICE_NAME: str = "Kalpi AI Portfolio Analyzer"

    # Agent (LLM is optional; rule-based fallback is used without a key).
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Server.
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

    # Market data / assumptions.
    BENCHMARK_TICKER: str = os.getenv("BENCHMARK_TICKER", "^NSEI")  # Nifty 50
    PRICE_HISTORY_PERIOD: str = os.getenv("PRICE_HISTORY_PERIOD", "1y")
    RISK_FREE_RATE: float = float(os.getenv("RISK_FREE_RATE", "0.065"))

    # CORS — allow local Streamlit/dev frontends.
    CORS_ORIGINS = _split_csv(os.getenv("CORS_ORIGINS", "")) or [
        "http://localhost:8501", "http://127.0.0.1:8501",
        "http://localhost:3000", "http://127.0.0.1:3000",
    ]


settings = Settings()
