# Kalpi AI Portfolio Analyzer

Turn a raw investment portfolio into a personalized, interactive, conversational
analysis. Upload or paste your holdings, ask questions in plain English, and watch
a dynamic canvas update with the relevant chart — performance, risk, diversification,
correlations, or a what-if simulation.

**Stack:** Pure Python · FastAPI backend · Streamlit "chat + canvas" frontend ·
yfinance for NSE/Nifty market data · an LLM-or-rules intent router.

---

## 1. Project overview

Kalpi AI Portfolio Analyzer is an MVP "AI portfolio advisor". A user provides a
portfolio of `Ticker, Weight` rows; an agent classifies each question into an
intent and routes it to a deterministic Python tool that computes the answer. The
backend returns a short human-readable summary plus structured `chart_data`, and
the Streamlit frontend renders a side-by-side dashboard that adapts to the
conversation.

## 2. Problem statement

Most retail investors and wealth managers are drowning in data but starving for
insight. Traditional dashboards present a wall of charts and Greek metrics (α, β)
and leave users to figure out what it means for their capital. This project makes
the portfolio *conversational*: plain-English questions in, institutional-grade
metrics and adaptive visuals out — with the actual math handled by trustworthy,
testable Python rather than a language model.

## 3. Features implemented

- **Zero-friction ingestion** — upload a CSV or paste `Ticker, Weight` text; weights
  are validated and normalized to sum to 1.
- **Chat + dynamic canvas** — a clean chat on the left; a canvas on the right that
  re-renders based on the intent of the latest message.
- **Return performance** — historical/annualized return, volatility, Sharpe ratio,
  and a Nifty 50 benchmark comparison.
- **Risk & vulnerabilities** — annualized volatility, maximum drawdown, historical
  VaR & CVaR (95%), and downside deviation.
- **Diversification & overlaps** — sector exposure, concentration (max weight, top-3,
  HHI), and a holdings correlation matrix / weighted-average correlation.
- **Portfolio summary & holding lookup** — instant overview and single-holding weight
  questions (no market-data round-trip needed).
- **What-if simulation** — exit a holding and reallocate (e.g. *"exit my largest
  position and move it to gold"*); refuses to guess partial moves.
- **Proactive smart prompts** — context-aware suggestions after a portfolio loads.
- **Works offline-by-default for the agent** — runs with no API key via a rule-based
  intent fallback.

## 4. Architecture (text diagram)

```
┌──────────────────────────────┐   HTTP/JSON   ┌──────────────────────────────────────────────┐
│  Streamlit Frontend          │ ────────────▶ │  FastAPI Backend                              │
│  chat + dynamic canvas       │               │  GET /health  POST /parse_portfolio  POST /chat│
│  session_state:              │ ◀──────────── │                                                │
│   portfolio, messages,       │   answer +    │   ┌────────────────────────────────────────┐  │
│   last_response, period      │   chart_data  │   │ Agent · orchestrator.py                 │  │
└──────────────────────────────┘               │   │  classify_intent → LLM label OR rules   │  │
                                                │   └──────────────────┬─────────────────────┘  │
                                                │                      │ intent                  │
                                                │   ┌──────────────────▼─────────────────────┐  │
                                                │   │ Tools · tools.py (deterministic)        │  │
                                                │   │  run_*_analysis → JSON-serializable dict │  │
                                                │   └──────────────────┬─────────────────────┘  │
                                                │                      │ calls                   │
                                                │   ┌──────────────────▼──────────┐ ┌──────────┐ │
                                                │   │ Core · pure Python math      │ │ Data      │ │
                                                │   │ performance / risk /         │ │ yfinance  │ │
                                                │   │ diversification / simulation │ │ (NSE/Nifty)│ │
                                                │   └──────────────────────────────┘ └──────────┘ │
                                                └──────────────────────────────────────────────┘
```

Data flow for a question: **message → classify intent → fetch prices (if needed) →
deterministic tool → format answer + chart_data → render canvas.**

## 5. The Golden Rule

> **The LLM never performs financial calculations or guesses metrics.**

The language model (when an API key is present) is used **only** to classify a
message into one of the supported intents — a single label word. Every number the
user sees is produced by deterministic Python functions in `backend/core/` and is
covered by unit tests. If no `OPENAI_API_KEY` is set, or the LLM call fails, a
rule-based keyword classifier takes over, so behaviour is reproducible and the app
runs fully offline. The agent's job is orchestration: *classify intent → call a
tool → format the tool's numbers.*

## 6. Agent / tool design

The agent classifies intent, then calls exactly one deterministic tool
(`backend/agent/tools.py`). Each tool composes pure functions from `backend/core/`
and returns a JSON-serializable dict.

| Intent | Tool | What it computes (deterministically) |
|---|---|---|
| `summary` | `run_summary_analysis` | # holdings, largest position, top-3 concentration, HHI, largest sector exposure, allocation (no market data) |
| `performance` | `run_performance_analysis` | total/annualized return, volatility, Sharpe, cumulative-return series, Nifty 50 benchmark comparison |
| `risk` | `run_risk_analysis` | annualized volatility, max drawdown, historical VaR & CVaR (95%), downside deviation, drawdown series |
| `diversification` | `run_diversification_analysis` | sector exposure, concentration metrics, correlation matrix, weighted-average correlation |
| `correlation` | `run_correlation_analysis` | holdings correlation matrix + weighted-average pairwise correlation |
| `holding_lookup` | `run_holding_lookup_analysis` | weight of a single holding, matched by ticker or common name alias (no market data) |
| `what_if` | `run_what_if_analysis` (`simulate_what_if`) | before/after allocation for an exit-and-reallocate edit, with before/after metric comparison |
| `general` | — | returns a helpful fallback + suggested questions |

Intent routing (`backend/agent/orchestrator.py`) uses an LLM label when available
and a keyword classifier otherwise, with a specificity-based tie-break
(`what_if > summary > holding_lookup > correlation > risk > performance >
diversification`). The response always reports which path ran (`method: "llm"` or
`"rules"`) and a confidence score.

## 7. Folder structure

```
Kalpi AI Portfolio Analyzer/
├── backend/
│   ├── main.py                 FastAPI app & routes
│   ├── config.py               settings from .env (CORS, benchmark, risk-free rate)
│   ├── agent/
│   │   ├── orchestrator.py      intent classification (LLM + rule fallback) — NO math
│   │   └── tools.py             deterministic tool wrappers + sector map + name aliases
│   ├── core/
│   │   ├── data.py              yfinance adjusted-close fetch (NSE .NS, Nifty ^NSEI)
│   │   ├── portfolio.py         parse / validate / normalize portfolios
│   │   ├── performance.py       returns, annualized return/vol, Sharpe, benchmark
│   │   ├── risk.py              drawdown, max drawdown, VaR, CVaR, downside deviation
│   │   ├── diversification.py   correlation, concentration (HHI), sector exposure
│   │   └── simulation.py        deterministic what-if (exit & reallocate)
│   └── models/
│       └── schemas.py          Pydantic models + API request bodies
├── frontend/
│   └── app.py                  Streamlit chat + dynamic canvas
├── data/
│   └── sample_portfolio.csv    example Indian-market portfolio
├── tests/                      offline unit + API tests (no live Yahoo Finance)
├── requirements.txt
├── .env.example
└── README.md
```

## 8. Setup

Requires Python 3.10+.

```bash
# from the project root
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # optional — the app runs without any API key
```

`.env` is optional. Set `OPENAI_API_KEY` to enable LLM-based intent classification;
without it, the deterministic rule-based router is used. Other settings (benchmark
ticker `^NSEI`, risk-free rate `0.065`, default period `1y`, CORS origins) have
sensible defaults.

## 9. Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Backend: http://localhost:8000 · interactive docs: http://localhost:8000/docs

## 10. Run the frontend

```bash
# in a second terminal, same virtualenv
streamlit run frontend/app.py
```

Frontend: http://localhost:8501 (expects the backend at `http://localhost:8000`;
override with the `KALPI_BACKEND_URL` env var).

## 11. API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/parse_portfolio` | Parse/validate/normalize a pasted CSV-like portfolio |
| `POST` | `/chat` | Classify intent, run the matching tool, return answer + chart data |

**`/chat` response shape:**

```json
{
  "intent": "risk",
  "classification": {"intent": "risk", "confidence": 0.8, "method": "rules"},
  "answer": "Annualized volatility is 18.4%, with a maximum drawdown of -12.1% ...",
  "metrics": { "...": "scalar metrics" },
  "chart_data": { "type": "risk", "drawdown": { "...": "date: value" } },
  "warnings": []
}
```

## 12. Example curl commands

Note: `/chat` takes a **`portfolio`** array of `{ticker, weight}` (not `holdings`).

```bash
# Health
curl http://localhost:8000/health
# -> {"status":"ok","service":"Kalpi AI Portfolio Analyzer"}

# Parse a pasted portfolio
curl -X POST http://localhost:8000/parse_portfolio \
  -H "Content-Type: application/json" \
  -d '{"text":"Ticker,Weight\nRELIANCE.NS,25\nTCS.NS,20"}'
# -> {"portfolio":[{"ticker":"RELIANCE.NS","weight":0.555...}, ...],
#     "total_weight":1.0,"message":"Portfolio parsed successfully"}

# Ask a question (uses the "portfolio" field)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
        "message": "What is my portfolio risk?",
        "portfolio": [
          {"ticker": "RELIANCE.NS", "weight": 0.25},
          {"ticker": "TCS.NS", "weight": 0.20},
          {"ticker": "INFY.NS", "weight": 0.15}
        ],
        "period": "1y"
      }'
```

## 13. Sample portfolio CSV format

Two columns — `Ticker` and `Weight` (case-insensitive; `Symbol`/`Quantity` also
accepted). NSE tickers use the `.NS` suffix. Weights may be percentages or
fractions; they are normalized to sum to 1. See `data/sample_portfolio.csv`:

```csv
Ticker,Weight
RELIANCE.NS,25
TCS.NS,20
HDFCBANK.NS,20
INFY.NS,15
ITC.NS,10
SUNPHARMA.NS,10
```

## 14. Example chat questions

- **Summary:** "Summarize my portfolio" · "What are my top holdings?"
- **Holding lookup:** "What is the weight of Reliance?" · "How much do I have in TCS?"
- **Performance:** "How has my portfolio performed versus the Nifty 50?" · "What is my Sharpe ratio?"
- **Risk:** "What is my risk — volatility, drawdown and VaR?"
- **Diversification:** "Am I over-concentrated in any sector?"
- **Correlation:** "Show me the correlation matrix of my holdings."
- **What-if:** "What if I exit my largest position and move it to gold?" ·
  "What if I exit TCS and allocate it to Gold?"

## 15. Portfolio state in Streamlit session state

The frontend holds all conversational state in `st.session_state`, so it persists
across reruns within a browser session:

- `portfolio` — the current normalized holdings (`[{ticker, weight}, ...]`), set
  after a successful `/parse_portfolio` call.
- `messages` — the running chat history (`{role, content}`), rendered as chat bubbles.
- `last_response` — the most recent `/chat` response; drives which canvas view renders.
- `period` — the selected price-history window (`1mo, 3mo, 6mo, 1y, 2y`; default `1y`).

Every `/chat` request sends the **current** `portfolio` and `period` from session
state, so the backend stays stateless — the conversation's portfolio context lives
in the session and is replayed on each turn. This is what lets what-if edits and
follow-up questions stay consistent mid-conversation without a database.

## 16. Smart prompts (proactive interaction)

After a portfolio loads, the frontend surfaces 2–3 context-aware prompts:

- If the **largest sector exposure ≥ 40%**: *"I noticed X% of your portfolio is in
  &lt;sector&gt;. See your sector exposure?"*
- If the **top-3 holdings ≥ 60%**: *"Your top 3 holdings make up X%. See
  concentration risk?"*
- Always: *"Would you like to compare your portfolio against Nifty 50?"*

Each prompt is a button that sends a concrete question into the chat, nudging the
user toward the vulnerabilities in their specific portfolio.

## 17. What-if simulation

`simulate_what_if` (in `backend/core/simulation.py`) applies a deterministic
*exit-and-reallocate* edit: it sets the exited holding's weight to 0, moves that
weight to the target (adding it as a new holding if needed), and re-normalizes.
Instructions are parsed deterministically (no LLM):

- Names/tickers are matched against the portfolio and a small alias map
  (e.g. *gold → GOLDBEES.NS*).
- *"my largest position"* resolves to the max-weight holding.
- **Partial moves are not guessed.** A request like *"reduce TCS by 10% and add it
  to HDFCBANK"* returns a clarification — *"Please confirm how much weight you want
  to move from TCS.NS to HDFCBANK.NS."* — rather than inventing an amount.

The tool returns before/after allocations and, when price data is available,
before/after performance metrics for comparison.

## 18. Tests

All tests are **offline** — they use synthetic price data and never call live Yahoo
Finance, so they're fast and deterministic.

```bash
pytest -q
```

Coverage spans the deterministic metric functions (performance, risk,
diversification), portfolio parsing/validation, the what-if simulation, intent
routing (rules + mocked LLM), the tool wrappers, and the FastAPI routes.

## 19. Known limitations

- **Market data / demo:** prices come from yfinance (NSE `.NS`, Nifty `^NSEI`), which
  can be rate-limited or temporarily unavailable; sector exposure uses a curated
  offline map for common Indian tickers (unmapped tickers show as "Unknown").
- **Simplified factor analysis:** "diversification" covers sector, concentration, and
  correlation — it does not implement full equity factor decomposition (momentum,
  value, size, etc.).
- **No persistent accounts:** state lives in the Streamlit session only; there is no
  database, login, or saved history.
- **Common-case what-if parsing:** natural-language what-ifs support exit-and-reallocate
  phrasings and "largest position"; partial/percentage moves ask for confirmation
  rather than guessing.
- **Not financial advice:** this is an educational prototype. Outputs are informational
  and must not be used as the basis for investment decisions.

## 20. Loom walkthrough script (3–5 min)

1. **Intro (20s).** "Kalpi AI Portfolio Analyzer — a conversational portfolio advisor.
   Pure Python, FastAPI + Streamlit. The key idea: the LLM only routes intent; all
   numbers come from deterministic, tested Python tools."
2. **Load a portfolio (30s).** Click **Load sample** (or paste `Ticker,Weight`). Show
   the parsed table and the **smart prompts** that appear based on the portfolio.
3. **Summary (20s).** Ask *"Summarize my portfolio."* — note it answers instantly
   (no market-data call) with holdings, largest position, top-3, and top sector.
4. **Performance (40s).** Ask *"How has my portfolio performed versus the Nifty 50?"* —
   show the cumulative-return chart, Sharpe/return/volatility cards, and benchmark
   comparison. Mention the detected intent / method caption.
5. **Risk (30s).** Ask *"What is my risk — volatility, drawdown and VaR?"* — the canvas
   swaps to the drawdown chart and risk cards.
6. **Diversification & correlation (30s).** Ask *"Am I over-concentrated in any sector?"*
   then *"Show me the correlation matrix."* — sector pie, concentration, heatmap.
7. **What-if mid-chat (40s).** Ask *"What if I exit my largest position and move it to
   gold?"* — show before/after allocation. Then *"reduce TCS by 10% and add it to
   HDFCBANK"* — show it asks for confirmation instead of guessing.
8. **The Golden Rule (20s).** Briefly show `backend/agent/` vs `backend/core/`: the
   agent classifies intent; the core computes every metric. Run `pytest -q` to show
   the deterministic functions are tested.
9. **Wrap (10s).** Recap: conversational, adaptive canvas, deterministic and testable.

---

*Built for the Kalpi Capital Engineering & Product assignment. Not financial advice.*
