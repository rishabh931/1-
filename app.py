import os
import json
import time
import math
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import streamlit as st

# ---------- Gemini ----------
try:
    import google.generativeai as genai
except Exception as e:
    genai = None

# ----------------- CONFIG -----------------
st.set_page_config(page_title="AI Stock Analyzer (Gemini)", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    .metric-card {border-radius:18px; padding:16px; background:rgba(255,255,255,0.65); box-shadow:0 6px 22px rgba(0,0,0,.06); border:1px solid rgba(0,0,0,.06)}
    .badge {display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#eef; margin-left:6px}
    .section {border-radius:22px; padding:18px 18px 8px; background:rgba(245,247,250,.75); border:1px solid rgba(0,0,0,.04); box-shadow:0 6px 22px rgba(0,0,0,.04)}
    .pill {font-size:12px; background:#f1f5ff; border:1px solid #dfe7ff; padding:5px 10px; border-radius:999px; margin-right:6px}
    .muted {color:#666}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HELPERS ----------
def resolve_symbol(user_text: str) -> str:
    """Accepts name or bare symbol and tries common Yahoo suffixes for India/US."""
    t = (user_text or "").strip().upper()
    if not t:
        return ""
    # If user gave explicit suffix, use as-is
    if "." in t:
        return t
    # Try common exchanges quickly
    for suf in [".NS", ".BO", ""]:
        sym = t + suf
        info = yf.Ticker(sym).fast_info if hasattr(yf.Ticker(sym), "fast_info") else {}
        try:
            _ = yf.Ticker(sym).info  # forces network call
            # If shortName or longName exists, consider valid
            if _ and ("shortName" in _ or "longName" in _):
                return sym
        except Exception:
            pass
    return t  # last resort; yfinance will error downstream

@st.cache_data(show_spinner=False, ttl=60*15)
def fetch_all(ticker: str):
    tk = yf.Ticker(ticker)
    # Price history
    hist = tk.history(period="5y", interval="1d", auto_adjust=True)
    # Financials
    is_y = tk.financials  # yearly income statement (wide-form)
    is_q = tk.quarterly_financials
    bs_y = tk.balance_sheet
    cf_y = tk.cashflow
    earnings = tk.earnings  # revenue & earnings year-wise
    info = {}
    try:
        info = tk.info
    except Exception:
        pass
    return hist, is_y, is_q, bs_y, cf_y, earnings, info

def calc_kpis(hist, is_y, is_q, info):
    last_price = float(hist["Close"].iloc[-1]) if len(hist) else np.nan
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else np.nan
    day_chg = (last_price - prev_close) / prev_close * 100 if prev_close and not math.isnan(prev_close) else np.nan

    mcap = info.get("marketCap")
    sector = info.get("sector") or "-"
    short_name = info.get("shortName") or info.get("longName") or "-"
    currency = info.get("currency") or "USD"

    # Pull revenue/OP/NI (best-effort from yearly income statement)
    def _get_latest(col, frame):
        try:
            return float(frame.loc[col].dropna().iloc[0])
        except Exception:
            return np.nan

    revenue = _get_latest("Total Revenue", is_y)
    operating_income = _get_latest("Operating Income", is_y)
    net_income = _get_latest("Net Income", is_y)

    opm = (operating_income / revenue * 100) if (revenue and not math.isnan(revenue) and revenue != 0) else np.nan
    npm = (net_income / revenue * 100) if (revenue and not math.isnan(revenue) and revenue != 0) else np.nan

    return {
        "last_price": last_price,
        "day_chg": day_chg,
        "mcap": mcap,
        "sector": sector,
        "name": short_name,
        "currency": currency,
        "revenue": revenue,
        "operating_income": operating_income,
        "net_income": net_income,
        "opm": opm,
        "npm": npm,
    }

def format_currency(x, currency="INR"):
    if pd.isna(x):
        return "—"
    suffixes = ["", "K", "M", "B", "T"]
    n = 0
    v = float(x)
    while abs(v) >= 1000 and n < len(suffixes)-1:
        v /= 1000.0
        n += 1
    return f"{currency} {v:,.2f}{suffixes[n]}"

def make_eps_df(is_q, info, hist):
    # Try quarterly EPS using diluted EPS from info (many tickers lack this; fallback to earnings growth proxy)
    # yfinance gives tk.quarterly_earnings sometimes
    eps_df = None
    try:
        qe = yf.Ticker(info.get("symbol", "")).quarterly_earnings
        if qe is not None and len(qe):
            eps_df = qe.reset_index().rename(columns={"Earnings": "EPS", "Quarter": "Period"})
    except Exception:
        pass

    if eps_df is None or eps_df.empty:
        # Dummy EPS growth proxy from net income / rolling shares (not reliable but shows a trend)
        try:
            close = hist["Close"].resample("Q").last()
            ret = close.pct_change().dropna()
            eps_df = pd.DataFrame({"Period": ret.index, "EPS": (1 + ret).cumprod()})
        except Exception:
            eps_df = pd.DataFrame(columns=["Period", "EPS"])
    return eps_df

def year_series_from_income(is_y, label):
    if is_y is None or is_y.empty or label not in is_y.index:
        return pd.DataFrame(columns=["Year", label])
    s = is_y.loc[label].dropna()
    df = pd.DataFrame({ "Year": [d.year if hasattr(d, "year") else str(d) for d in s.index], label: s.values })
    return df

def prompt_for_gemini(company, ticker, kpis, is_y, is_q, eps_df):
    # Convert a compact snapshot for the model
    def df_to_records(df, maxrows=8):
        try:
            return df.tail(maxrows).to_dict(orient="records")
        except Exception:
            return []

    revenue_df = year_series_from_income(is_y, "Total Revenue")
    op_df = year_series_from_income(is_y, "Operating Income")
    ni_df = year_series_from_income(is_y, "Net Income")

    data_snapshot = {
        "company": company,
        "ticker": ticker,
        "currency": kpis["currency"],
        "metrics": {
            "last_price": kpis["last_price"],
            "day_change_pct": kpis["day_chg"],
            "market_cap": kpis["mcap"],
            "revenue_latest": kpis["revenue"],
            "operating_income_latest": kpis["operating_income"],
            "net_income_latest": kpis["net_income"],
            "opm_pct_latest": kpis["opm"],
            "npm_pct_latest": kpis["npm"],
        },
        "revenue_series": df_to_records(revenue_df),
        "operating_income_series": df_to_records(op_df),
        "net_income_series": df_to_records(ni_df),
        "eps_series": df_to_records(eps_df)
    }

    system = """
You are a buy-side equity analyst. Write concise, **bullet-point** insights from fundamentals and trends.
Respond STRICTLY in JSON with the following top-level keys:

{
 "revenue_profit": {
   "summary": "1-2 line narrative",
   "bullets": ["...", "..."]
 },
 "profitability": {
   "summary": "...",
   "bullets": ["...", "..."]
 },
 "eps_trend": {
   "summary": "...",
   "bullets": ["...", "..."]
 },
 "investment_sentiment": {
   "summary": "...",
   "bullets": ["...", "..."],
   "verdict": "Bullish | Neutral | Cautious | Bearish",
   "confidence_pct": 0-100
 }
}

Guidelines:
- Quantify growth (CAGR/YoY) when possible using provided series.
- Highlight operating leverage, margin trajectory, and sustainability of growth.
- Avoid price targets or personalized financial advice.
- Keep each section to 4–6 bullets, crisp and factual.
"""
    user = {
        "task": "Generate structured insights",
        "data": data_snapshot
    }
    return system, user

def call_gemini(system_prompt, user_payload, api_key: str, model_name: str):
    if genai is None:
        raise RuntimeError("google-generativeai not installed.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    # Use JSON bias by strongly steering format
    prompt = f"{system_prompt}\n\nDATA (JSON):\n{json.dumps(user_payload, default=str)}"
    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    # Try extract json block
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end+1])
            return parsed, text
    except Exception:
        pass
    # Fallback empty structure
    return {
        "revenue_profit": {"summary":"", "bullets":[]},
        "profitability": {"summary":"", "bullets":[]},
        "eps_trend": {"summary":"", "bullets":[]},
        "investment_sentiment": {"summary":"", "bullets":[], "verdict":"Neutral", "confidence_pct":50},
    }, text

def kpi_card(label, value, sub=None):
    with st.container(border=False):
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**{label}**")
        st.markdown(f"<h3 style='margin:6px 0 2px 0'>{value}</h3>", unsafe_allow_html=True)
        if sub:
            st.markdown(f"<span class='muted'>{sub}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    default_model = "gemini-1.5-flash"
    model_name = st.selectbox("Gemini model", [default_model, "gemini-1.5-pro"], index=0)
    st.caption("Tip: Flash is fast/cheap for summaries; Pro is best for nuanced analysis.")

    api_key = st.text_input("Gemini API Key", value=st.secrets.get("GEMINI_API_KEY", ""), type="password")
    st.caption("Store securely in .streamlit/secrets.toml as GEMINI_API_KEY to auto-load.")

    st.divider()
    st.markdown("**Input can be:** `CDSL`, `TCS`, `INFY`, `AAPL`, etc. I’ll try `.NS/.BO` if needed.")
    user_text = st.text_input("Stock name or symbol", value="CDSL")
    run = st.button("Analyze", type="primary", use_container_width=True)

# --------------- MAIN -------------------
st.title("📊 AI Stock Analyzer — Gemini")
st.caption("Enter a stock → latest market & financial data → AI-driven insights with clean charts and sections.")

if run:
    with st.spinner("Resolving symbol & fetching data..."):
        sym = resolve_symbol(user_text)
        try:
            hist, is_y, is_q, bs_y, cf_y, earnings, info = fetch_all(sym)
        except Exception as e:
            st.error(f"Could not fetch data for `{sym}` — {e}")
            st.stop()

    kpis = calc_kpis(hist, is_y, is_q, info)
    company = kpis["name"]
    currency = kpis["currency"]

    # KPIs row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Last Price", format_currency(kpis["last_price"], currency), sub=f"{kpis['day_chg']:.2f}% today" if not pd.isna(kpis["day_chg"]) else "—")
    with c2:
        kpi_card("Market Cap", format_currency(kpis["mcap"], currency), sub=kpis["sector"])
    with c3:
        kpi_card("Operating Margin", f"{kpis['opm']:.1f}%" if not pd.isna(kpis["opm"]) else "—", sub="Latest FY")
    with c4:
        kpi_card("Net Profit Margin", f"{kpis['npm']:.1f}%" if not pd.isna(kpis["npm"]) else "—", sub="Latest FY")

    # Price chart
    if not hist.empty:
        st.markdown("#### Price Trend")
        fig = px.line(hist.reset_index(), x="Date", y="Close", title=None)
        fig.update_layout(margin=dict(l=10,r=10,b=10,t=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price history available.")

    # Build series & EPS df
    eps_df = make_eps_df(is_q, info, hist)

    # Revenue & OP chart
    rev_df = year_series_from_income(is_y, "Total Revenue")
    op_df = year_series_from_income(is_y, "Operating Income")
    if not rev_df.empty:
        st.markdown("#### Revenue & Operating Profit (FY)")
        m = rev_df.merge(op_df, on="Year", how="left")
        m = m.tail(6)
        m2 = m.melt(id_vars="Year", var_name="Metric", value_name="Value")
        fig2 = px.bar(m2, x="Year", y="Value", color="Metric", barmode="group")
        fig2.update_layout(margin=dict(l=10,r=10,b=10,t=10), height=360)
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- Gemini insights ----------
    if not api_key:
        st.warning("Add your Gemini API key in the sidebar to generate AI insights.")
        st.stop()

    with st.spinner("Generating AI insights (Gemini)…"):
        sys, user_payload = prompt_for_gemini(company, sym, kpis, is_y, is_q, eps_df)
        try:
            insights, raw_text = call_gemini(sys, user_payload, api_key, model_name)
        except Exception as e:
            st.error(f"Gemini error: {e}")
            st.stop()

    # --------- RENDER SECTIONS ----------
    def render_section(title, key):
        block = insights.get(key, {}) if isinstance(insights, dict) else {}
        st.markdown(f"### {title}")
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        if block.get("summary"):
            st.write(block["summary"])
        bullets = block.get("bullets", [])
        if bullets:
            st.markdown("\n".join([f"- {b}" for b in bullets]))
        st.markdown("</div>", unsafe_allow_html=True)

    render_section("Revenue & Operating Profit Analysis", "revenue_profit")
    render_section("Profitability Analysis", "profitability")

    # EPS Chart + bullets
    st.markdown("### EPS Performance Trend")
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    if not eps_df.empty:
        fig3 = px.line(eps_df, x=eps_df.columns[0], y=eps_df.columns[1], markers=True)
        fig3.update_layout(margin=dict(l=10,r=10,b=10,t=10), height=320)
        st.plotly_chart(fig3, use_container_width=True)
    block = insights.get("eps_trend", {})
    if block.get("summary"): st.write(block["summary"])
    if block.get("bullets"): st.markdown("\n".join([f"- {b}" for b in block["bullets"]]))
    st.markdown("</div>", unsafe_allow_html=True)

    # Investment Sentiment
    st.markdown("### Investment Sentiment & Key Insights")
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    inv = insights.get("investment_sentiment", {})
    if inv.get("summary"): st.write(inv["summary"])
    if inv.get("bullets"): st.markdown("\n".join([f"- {b}" for b in inv["bullets"]]))
    verdict = inv.get("verdict", "Neutral")
    conf = inv.get("confidence_pct", 50)
    st.markdown(
        f"**Verdict:** <span class='pill'>{verdict}</span>  **Confidence:** <span class='pill'>{conf}%</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(f"Data source: Yahoo Finance via yfinance • Ticker resolved: **{sym}** • Company: **{company}**")

else:
    st.info("Enter a stock name/symbol in the sidebar (e.g., **CDSL**, **TCS**, **INFY**, **AAPL**) and click **Analyze**.")
