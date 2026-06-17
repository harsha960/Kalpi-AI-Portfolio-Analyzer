"""Kalpi AI Portfolio Analyzer - Streamlit frontend (chat + dynamic canvas).

A thin UI over the FastAPI backend. The left column ingests a portfolio and
hosts the chat; the right column is a canvas that re-renders based on the intent
the backend returns. No financial math happens here - everything comes from the
backend's deterministic tools.

Run the backend first, then:  streamlit run frontend/app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BACKEND_URL = os.getenv("KALPI_BACKEND_URL", "http://localhost:8000")
PERIODS = ["1mo", "3mo", "6mo", "1y", "2y"]
SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_portfolio.csv"
BACKEND_DOWN_MSG = (
    "Backend is not running. Start it with:\n\n"
    "`uvicorn backend.main:app --reload --port 8000`"
)
SUGGESTED_QUESTIONS = [
    "How has my portfolio performed versus the Nifty 50?",
    "What is my portfolio's risk - volatility, drawdown and VaR?",
    "Am I over-concentrated in any sector?",
    "Show me the correlation matrix of my holdings.",
    "What if I exit my largest position and move it to gold?",
]


# --------------------------------------------------------------------------- #
# Backend client
# --------------------------------------------------------------------------- #
def api_get(path: str):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=10)
    except requests.exceptions.RequestException:
        return None, BACKEND_DOWN_MSG
    if r.status_code == 200:
        return r.json(), None
    return None, f"Error {r.status_code}: {r.text}"


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=90)
    except requests.exceptions.RequestException:
        return None, BACKEND_DOWN_MSG
    if r.status_code == 200:
        return r.json(), None
    try:
        detail = r.json().get("detail", r.text)
    except Exception:
        detail = r.text
    return None, f"Error {r.status_code}: {detail}"


def pct(value, dp: int = 1) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value) * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "n/a"


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
def parse_and_store(text: str):
    if not text or not text.strip():
        st.warning("Upload a CSV or paste your holdings first.")
        return
    resp, err = api_post("/parse_portfolio", {"text": text})
    if err:
        st.error(err)
        return
    st.session_state.portfolio = resp["portfolio"]
    st.session_state.last_response = None
    st.success(f"{resp['message']} ({len(resp['portfolio'])} holdings).")


def send_chat(message: str):
    st.session_state.messages.append({"role": "user", "content": message})
    with st.spinner("Analyzing..."):
        resp, err = api_post(
            "/chat",
            {
                "message": message,
                "portfolio": st.session_state.portfolio,
                "period": st.session_state.period,
            },
        )
    if err:
        st.session_state.messages.append({"role": "assistant", "content": err})
        st.session_state.last_response = None
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": resp.get("answer", "(no answer)")}
        )
        st.session_state.last_response = resp


# --------------------------------------------------------------------------- #
# Canvas renderers
# --------------------------------------------------------------------------- #
def render_welcome():
    st.info("Load a portfolio and ask a question to populate this canvas.")
    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        fig = px.pie(df, values="weight", names="ticker", hole=0.4, title="Current Allocation")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Try asking:**")
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if st.button(q, key=f"welcome_{i}", use_container_width=True):
            send_chat(q)
            st.rerun()


def render_performance(chart: dict, metrics: dict):
    cols = st.columns(4)
    cols[0].metric("Total Return", pct(metrics.get("total_return")))
    cols[1].metric("Annualized", pct(metrics.get("annualized_return")))
    cols[2].metric("Volatility", pct(metrics.get("annualized_volatility")))
    cols[3].metric("Sharpe", f"{metrics.get('sharpe_ratio', float('nan')):.2f}")

    cum = chart.get("cumulative_return", {})
    if cum:
        s = pd.Series(cum)
        s.index = pd.to_datetime(s.index)
        dfc = pd.DataFrame({"Date": s.index, "Cumulative Return %": s.values * 100})
        st.plotly_chart(px.line(dfc, x="Date", y="Cumulative Return %", title="Cumulative Return"),
                        use_container_width=True)
    else:
        st.info("No cumulative-return series available.")

    if "benchmark" in metrics:
        b = metrics["benchmark"]
        pf, bm = b["portfolio"], b["benchmark"]
        st.markdown(f"**Benchmark — {bm['name']}**")
        bc = st.columns(2)
        bc[0].metric("Portfolio Ann. Return", pct(pf["annualized_return"]))
        bc[1].metric(f"{bm['name']} Ann. Return", pct(bm["annualized_return"]),
                     delta=pct(b["excess_annualized_return"]))


def render_risk(chart: dict, metrics: dict):
    cols = st.columns(4)
    cols[0].metric("Volatility", pct(metrics.get("annualized_volatility")))
    cols[1].metric("Max Drawdown", pct(metrics.get("max_drawdown")))
    cols[2].metric("VaR 95% (1d)", pct(metrics.get("historical_var_95"), 2))
    cols[3].metric("CVaR 95% (1d)", pct(metrics.get("historical_cvar_95"), 2))
    st.metric("Downside Deviation", pct(metrics.get("downside_deviation")))

    dd = chart.get("drawdown", {})
    if dd:
        s = pd.Series(dd)
        s.index = pd.to_datetime(s.index)
        dfd = pd.DataFrame({"Date": s.index, "Drawdown %": s.values * 100})
        st.plotly_chart(px.area(dfd, x="Date", y="Drawdown %", title="Drawdown"),
                        use_container_width=True)
    else:
        st.info("No drawdown series available.")


def render_diversification(chart: dict, metrics: dict):
    c = metrics.get("concentration", {})
    sectors = metrics.get("sector_exposure", {}) or chart.get("sector_exposure", {})
    wavg = metrics.get("weighted_avg_correlation")

    cols = st.columns(4)
    cols[0].metric("Largest Position", str(c.get("largest_position", "-")))
    cols[1].metric("Max Weight", pct(c.get("max_weight")))
    cols[2].metric("Top 3 Weight", pct(c.get("top_3_weight")))
    cols[3].metric("HHI", f"{c.get('hhi', float('nan')):.2f}")

    if sectors:
        dfp = pd.DataFrame({"Sector": list(sectors.keys()), "Weight": list(sectors.values())})
        st.plotly_chart(px.pie(dfp, values="Weight", names="Sector", hole=0.4, title="Sector Exposure"),
                        use_container_width=True)
    else:
        st.info("No sector data available.")

    matrix = chart.get("correlation_matrix")
    if matrix:
        _heatmap(matrix)
    if wavg is not None:
        st.caption(f"Weighted-average pairwise correlation: {wavg:.2f}")


def render_correlation(chart: dict, metrics: dict):
    wavg = metrics.get("weighted_avg_correlation")
    if wavg is not None:
        st.metric("Weighted-Avg Correlation", f"{wavg:.2f}")
    matrix = chart.get("correlation_matrix")
    if matrix:
        _heatmap(matrix)
    else:
        st.info("No correlation data available.")


def render_what_if(chart: dict, metrics: dict):
    before = chart.get("before_portfolio", {})
    after = chart.get("after_portfolio", {})
    if before or after:
        rows = [{"Ticker": t, "Weight %": w * 100, "Version": "Before"} for t, w in before.items()]
        rows += [{"Ticker": t, "Weight %": w * 100, "Version": "After"} for t, w in after.items()]
        dfw = pd.DataFrame(rows)
        st.plotly_chart(
            px.bar(dfw, x="Ticker", y="Weight %", color="Version", barmode="group",
                   title="Before vs After Allocation"),
            use_container_width=True,
        )
    else:
        st.info("No simulation result to display.")

    b, a = metrics.get("before"), metrics.get("after")
    if b and a:
        cols = st.columns(2)
        cols[0].metric("Sharpe (after)", f"{a['sharpe_ratio']:.2f}",
                       delta=f"{a['sharpe_ratio'] - b['sharpe_ratio']:.2f}")
        cols[1].metric("Volatility (after)", pct(a["annualized_volatility"]),
                       delta=pct(a["annualized_volatility"] - b["annualized_volatility"]),
                       delta_color="inverse")


def render_general(resp: dict):
    suggestions = (resp.get("chart_data", {}) or {}).get("suggestions", SUGGESTED_QUESTIONS)
    st.markdown("**Suggested questions**")
    for i, q in enumerate(suggestions):
        if st.button(q, key=f"gen_{i}", use_container_width=True):
            send_chat(q)
            st.rerun()


def _heatmap(matrix: dict):
    df = pd.DataFrame(matrix)
    df = df.reindex(index=df.columns)  # keep rows/cols in the same order
    fig = px.imshow(df, text_auto=".2f", color_continuous_scale="RdBu",
                    zmin=-1, zmax=1, aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)


# Mirrors backend DEFAULT_SECTOR_MAP; used only to suggest smart prompts client-side.
FRONTEND_SECTOR_MAP = {
    "RELIANCE.NS": "Energy / Conglomerate", "TCS.NS": "Information Technology",
    "INFY.NS": "Information Technology", "WIPRO.NS": "Information Technology",
    "HCLTECH.NS": "Information Technology", "HDFCBANK.NS": "Financial Services",
    "ICICIBANK.NS": "Financial Services", "SBIN.NS": "Financial Services",
    "KOTAKBANK.NS": "Financial Services", "AXISBANK.NS": "Financial Services",
    "BAJFINANCE.NS": "Financial Services", "ITC.NS": "FMCG", "HINDUNILVR.NS": "FMCG",
    "NESTLEIND.NS": "FMCG", "SUNPHARMA.NS": "Healthcare", "DRREDDY.NS": "Healthcare",
    "LT.NS": "Industrials", "BHARTIARTL.NS": "Telecom", "MARUTI.NS": "Automobile",
    "TATAMOTORS.NS": "Automobile", "ASIANPAINT.NS": "Consumer Durables",
    "GOLDBEES.NS": "Gold / Commodity", "SILVERBEES.NS": "Gold / Commodity",
}


def render_summary(chart: dict, metrics: dict):
    c = metrics.get("concentration", {})
    sectors = metrics.get("sector_exposure", {}) or chart.get("sector_exposure", {})
    cols = st.columns(4)
    cols[0].metric("Holdings", metrics.get("num_holdings", "-"))
    cols[1].metric("Largest", str(c.get("largest_position", "-")))
    cols[2].metric("Top 3 Weight", pct(c.get("top_3_weight")))
    cols[3].metric("Top Sector", next(iter(sectors)) if sectors else "-")
    alloc = chart.get("allocation", {})
    if alloc:
        dfa = pd.DataFrame({"Ticker": list(alloc), "Weight": list(alloc.values())})
        st.plotly_chart(px.pie(dfa, values="Weight", names="Ticker", hole=0.4,
                               title="Current Allocation"), use_container_width=True)


def render_holding_lookup(chart: dict, metrics: dict):
    ticker = metrics.get("ticker")
    if metrics.get("found") and ticker is not None:
        st.metric(str(ticker), pct(metrics.get("weight")))
    else:
        st.info("That holding is not in your portfolio.")
    alloc = chart.get("allocation", {})
    if alloc:
        dfa = pd.DataFrame({"Ticker": list(alloc), "Weight": [w * 100 for w in alloc.values()]})
        dfa["Holding"] = dfa["Ticker"].apply(lambda t: "Selected" if t == ticker else "Other")
        st.plotly_chart(px.bar(dfa, x="Ticker", y="Weight", color="Holding",
                               title="Allocation"), use_container_width=True)


def _smart_prompts():
    pf = st.session_state.portfolio
    if not pf:
        return []
    weights = sorted((h["weight"] for h in pf), reverse=True)
    top3 = sum(weights[:3])
    sectors = {}
    for h in pf:
        sector = FRONTEND_SECTOR_MAP.get(h["ticker"], "Unknown")
        if sector != "Unknown":
            sectors[sector] = sectors.get(sector, 0.0) + h["weight"]
    prompts = []
    if sectors:
        top_sector = max(sectors, key=sectors.get)
        if sectors[top_sector] >= 0.40:
            prompts.append((
                f"I noticed {sectors[top_sector] * 100:.0f}% of your portfolio is in "
                f"{top_sector}. See your sector exposure?",
                "What is my sector exposure and concentration?",
            ))
    if top3 >= 0.60:
        prompts.append((
            f"Your top 3 holdings make up {top3 * 100:.0f}%. See concentration risk?",
            "What is my portfolio risk - volatility, drawdown and VaR?",
        ))
    prompts.append((
        "Would you like to compare your portfolio against Nifty 50?",
        "How has my portfolio performed versus the Nifty 50?",
    ))
    return prompts


def render_canvas():
    resp = st.session_state.last_response
    if resp is None:
        render_welcome()
        return

    if (resp.get("chart_data", {}) or {}).get("type") == "need_portfolio":
        st.info("Load a portfolio (left) to see this analysis.")
        return

    for w in resp.get("warnings", []):
        st.warning(w)

    cls = resp.get("classification", {})
    st.caption(
        f"Detected intent: **{resp.get('intent')}** · via {cls.get('method')} "
        f"· confidence {cls.get('confidence')}"
    )

    intent = resp.get("intent", "general")
    chart = resp.get("chart_data", {}) or {}
    metrics = resp.get("metrics", {}) or {}
    dispatch = {
        "performance": render_performance,
        "risk": render_risk,
        "diversification": render_diversification,
        "correlation": render_correlation,
        "what_if": render_what_if,
        "summary": render_summary,
        "holding_lookup": render_holding_lookup,
    }
    if intent in dispatch:
        dispatch[intent](chart, metrics)
    else:
        render_general(resp)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Kalpi AI Portfolio Analyzer", page_icon="📊", layout="wide")

st.session_state.setdefault("portfolio", [])
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_response", None)
st.session_state.setdefault("period", "1y")

st.title("📊 Kalpi AI Portfolio Analyzer")
st.caption(
    "LLM is used only for intent routing. All financial metrics are computed by "
    "deterministic Python tools."
)

_, health_err = api_get("/health")
if health_err:
    st.error(health_err)

left, right = st.columns([1, 1.1], gap="large")

# ----------------------------- Left: ingestion + chat ----------------------- #
with left:
    st.subheader("Portfolio")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    pasted = st.text_area(
        "...or paste holdings (Ticker, Weight per line)",
        height=120,
        placeholder="Ticker,Weight\nRELIANCE.NS,25\nTCS.NS,20",
    )
    b1, b2 = st.columns(2)
    if b1.button("Parse portfolio", use_container_width=True):
        if uploaded is not None:
            parse_and_store(uploaded.getvalue().decode("utf-8", errors="replace"))
        else:
            parse_and_store(pasted)
    if b2.button("Load sample", use_container_width=True):
        try:
            parse_and_store(SAMPLE_CSV.read_text())
        except Exception as exc:
            st.error(f"Could not read sample portfolio: {exc}")

    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        df["Weight %"] = (df["weight"] * 100).round(2)
        st.dataframe(df.rename(columns={"ticker": "Ticker"})[["Ticker", "Weight %"]],
                     hide_index=True, use_container_width=True)
    else:
        st.info("No portfolio loaded yet.")

    prompts = _smart_prompts()
    if prompts:
        st.markdown("**Smart prompts**")
        for i, (label, query) in enumerate(prompts):
            if st.button(label, key=f"smart_{i}", use_container_width=True):
                send_chat(query)
                st.rerun()

    st.selectbox("Price-history period", PERIODS, key="period")

    st.divider()
    st.subheader("Chat")
    if not st.session_state.portfolio:
        st.warning("Load a portfolio to ask analytical questions (general questions still work).")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input("Ask a question", placeholder="e.g. What is my portfolio risk?")
        sent = st.form_submit_button("Send", use_container_width=True)
    if sent and user_msg.strip():
        send_chat(user_msg.strip())
        st.rerun()

# ----------------------------- Right: dynamic canvas ------------------------ #
with right:
    st.subheader("Canvas")
    render_canvas()
