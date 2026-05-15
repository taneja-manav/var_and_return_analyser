"""
VaR Dashboard — Streamlit App
==============================
Interactive Value at Risk analysis using Historical, Parametric & Monte Carlo methods.
Users enter stock tickers and adjust confidence level via a slider.
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="VaR Dashboard", page_icon="📉", layout="wide")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: red;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        color: white;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.65; }
    .metric-card .value { font-size: 1.65rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-card .sub   { font-size: 0.82rem; opacity: 0.7; }

    .var-negative { color: #ff6b6b; }
    .var-positive { color: #51cf66; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027, #203a43);
    }
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] span { color: #e0e0e0 !important; }

    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #2c5364, transparent);
        border: none;
        margin: 2rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📉 Value at Risk Dashboard(what's at risk)</h1>
    <p>Historical · Parametric · Monte Carlo — compare VaR across stocks and confidence levels,compare diffrent stocks and make the right choice to park your surplus</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    tickers_input = st.text_input(
        "Stock Tickers (comma-separated)",
        value="",
        placeholder="e.g. AAPL, MSFT, 2222.SR, RELIANCE.NS",
        help="Yahoo Finance tickers, e.g. AAPL, MSFT, 2222.SR",
    )

    confidence = st.slider(
        "Confidence Level (%)",
        min_value=90, max_value=99, value=95, step=1,
        help="Higher confidence → larger VaR estimate",
    )
    conf = confidence / 100.0

    period = st.selectbox("Historical Period", ["1y", "2y", "3y", "5y", "10y"], index=3)

    investment = st.number_input(
        "Portfolio Value ($)", min_value=1_000, value=1_000_000, step=10_000, format="%d"
    )

    mc_sims = st.select_slider(
        "Monte Carlo Simulations",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
    )

    run_btn = st.button("🚀 Calculate VaR", use_container_width=True, type="primary")

# ─── VaR Functions ────────────────────────────────────────────────────────────

def historical_var(ret, confidence):
    return np.percentile(ret, (1 - confidence) * 100)

def parametric_var(ret, confidence):
    mu, sigma = ret.mean(), ret.std()
    return mu + norm.ppf(1 - confidence) * sigma

def monte_carlo_var(ret, confidence, n_sims):
    mu, sigma = ret.mean(), ret.std()
    simulated = np.random.normal(mu, sigma, n_sims)
    return np.percentile(simulated, (1 - confidence) * 100)

# ─── Main Logic ───────────────────────────────────────────────────────────────
if run_btn:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) < 1:
        st.error("Please enter at least one ticker.")
        st.stop()

    # Download
    with st.spinner("📥 Downloading price data …"):
        try:
            raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
            if raw.empty:
                st.error("No data returned. Check your tickers and try again.")
                st.stop()
            prices = raw["Close"] if len(tickers) > 1 else raw[["Close"]].rename(columns={"Close": tickers[0]})
            prices.columns = tickers
            returns = prices.pct_change().dropna()
        except Exception as e:
            st.error(f"Download error: {e}")
            st.stop()

    st.markdown(f"**Data range:** `{returns.index[0].date()}` → `{returns.index[-1].date()}`  &nbsp;|&nbsp;  **{len(returns):,}** trading days")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Compute VaR ───────────────────────────────────────────────────────
    results = {}
    for tkr in tickers:
        r = returns[tkr].dropna().values
        results[tkr] = {
            "Historical":  historical_var(r, conf),
            "Parametric":  parametric_var(r, conf),
            "Monte Carlo": monte_carlo_var(r, conf, mc_sims),
            "returns": r,
            "mu": r.mean(),
            "sigma": r.std(),
        }

    # ── Metric Cards ──────────────────────────────────────────────────────
    for tkr in tickers:
        st.markdown(f"### 🏷️ {tkr}")
        cols = st.columns(3)
        for i, method in enumerate(["Historical", "Parametric", "Monte Carlo"]):
            var_pct = results[tkr][method]
            var_dollar = abs(var_pct) * investment
            css_class = "var-negative" if var_pct < 0 else "var-positive"
            cols[i].markdown(f"""
            <div class="metric-card">
                <div class="label">{method} VaR</div>
                <div class="value {css_class}">{var_pct:.4%}</div>
                <div class="sub">${var_dollar:,.0f} at risk</div>
            </div>
            """, unsafe_allow_html=True)
        st.write("")  # spacer

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Distribution Plots ────────────────────────────────────────────────
    palette = [
        "#00b4d8", "#e63946", "#2a9d8f", "#f4a261",
        "#7209b7", "#3a86a7", "#06d6a0", "#ef476f",
    ]

    st.markdown("## 📊 Return Distributions & VaR Thresholds")

    for idx, tkr in enumerate(tickers):
        r = results[tkr]["returns"]
        mu, sigma = results[tkr]["mu"], results[tkr]["sigma"]
        color = palette[idx % len(palette)]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Historical VaR", "Parametric VaR", "Monte Carlo VaR"),
            horizontal_spacing=0.07,
        )

        # (a) Historical
        h_var = results[tkr]["Historical"]
        fig.add_trace(go.Histogram(
            x=r, nbinsx=80, histnorm="probability density",
            marker_color=color, opacity=0.7, name="Actual Returns",
            showlegend=False,
        ), row=1, col=1)
        fig.add_vline(x=h_var, line_dash="dash", line_color="#ff6b6b", line_width=2, row=1, col=1,
                      annotation_text=f"VaR {confidence}% = {h_var:.4f}", annotation_font_color="#ff6b6b")

        # (b) Parametric
        p_var = results[tkr]["Parametric"]
        x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        pdf = norm.pdf(x_range, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x_range, y=pdf, mode="lines",
            line=dict(color=color, width=2.5), name="Normal PDF",
            showlegend=False, fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
        ), row=1, col=2)
        # shade tail
        tail_x = x_range[x_range <= p_var]
        tail_y = norm.pdf(tail_x, mu, sigma)
        fig.add_trace(go.Scatter(
            x=np.concatenate([tail_x, tail_x[::-1]]),
            y=np.concatenate([tail_y, np.zeros_like(tail_y)]),
            fill="toself", fillcolor="rgba(255,107,107,0.35)",
            line=dict(width=0), showlegend=False,
        ), row=1, col=2)
        fig.add_vline(x=p_var, line_dash="dash", line_color="#ff6b6b", line_width=2, row=1, col=2,
                      annotation_text=f"VaR {confidence}% = {p_var:.4f}", annotation_font_color="#ff6b6b")

        # (c) Monte Carlo
        mc_var = results[tkr]["Monte Carlo"]
        simulated = np.random.normal(mu, sigma, mc_sims)
        fig.add_trace(go.Histogram(
            x=simulated, nbinsx=80, histnorm="probability density",
            marker_color=color, opacity=0.7, name="Simulated Returns",
            showlegend=False,
        ), row=1, col=3)
        fig.add_vline(x=mc_var, line_dash="dash", line_color="#ff6b6b", line_width=2, row=1, col=3,
                      annotation_text=f"VaR {confidence}% = {mc_var:.4f}", annotation_font_color="#ff6b6b")

        fig.update_layout(
            title=dict(text=f"<b>{tkr}</b>", font_size=16),
            height=380,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,35,0.6)",
            margin=dict(t=60, b=40, l=50, r=30),
        )
        fig.update_xaxes(title_text="Daily Return")
        fig.update_yaxes(title_text="Density", col=1)

        st.plotly_chart(fig, use_container_width=True)

    # ── Comparison Bar Chart ──────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 VaR Comparison")

    methods = ["Historical", "Parametric", "Monte Carlo"]
    comp_fig = go.Figure()

    for idx, tkr in enumerate(tickers):
        vals = [abs(results[tkr][m]) * 100 for m in methods]
        comp_fig.add_trace(go.Bar(
            name=tkr, x=methods, y=vals,
            marker_color=palette[idx % len(palette)],
            text=[f"{v:.2f}%" for v in vals],
            textposition="outside",
            textfont=dict(size=12, color="white"),
        ))

    comp_fig.update_layout(
        barmode="group",
        title=dict(text=f"<b>VaR Comparison ({confidence}% Confidence, 1-Day)</b>", font_size=16),
        yaxis_title="VaR (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.6)",
        height=450,
        margin=dict(t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(comp_fig, use_container_width=True)

    # ── Summary Table ─────────────────────────────────────────────────────
    st.markdown("## 📋 Summary Table")
    rows = []
    for tkr in tickers:
        for method in methods:
            var_val = results[tkr][method]
            rows.append({
                "Ticker": tkr,
                "Method": method,
                "VaR (%)": f"{var_val:.4%}",
                f"VaR (${investment:,.0f})": f"${abs(var_val) * investment:,.2f}",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

else:
    # Landing state
    st.info("👈 Enter stock tickers, adjust the confidence slider, and hit **Calculate VaR** to begin.")
    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, [
        ("📜", "Historical", "Uses the empirical percentile of actual past returns."),
        ("📐", "Parametric", "Assumes normal distribution — mean + z·σ approach."),
        ("🎲", "Monte Carlo", "Simulates thousands of random return scenarios."),
    ]):
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem">{icon}</div>
            <div class="label" style="margin:0.5rem 0;font-size:0.95rem;opacity:1;font-weight:600">{title}</div>
            <div class="sub">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
