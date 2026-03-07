#!/usr/bin/env python3
"""
D10 Strategy Dashboard
======================
Streamlit app — reads pre-computed data from data/ directory.
Data is refreshed daily by GitHub Actions running d10_daily_score.py.
"""

import os, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="D10 Strategy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    history = pd.read_csv(
        os.path.join(DATA_DIR, "combo_history.csv"),
        index_col=0, parse_dates=True
    )
    with open(os.path.join(DATA_DIR, "signals_today.json")) as f:
        today_data = json.load(f)
    with open(os.path.join(DATA_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return history, today_data, metrics

try:
    history, today_data, metrics = load_data()
except FileNotFoundError:
    st.error("No data found. Run `python d10_daily_score.py` first to generate data files.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("D10 Strategy Dashboard")
st.caption(f"Last updated: **{today_data['date']}** · 73 Signals · HO Pairs · HO Weights · 2Y+3Y+4Y Ensemble")

# ── Top metrics row ───────────────────────────────────────────────────────────
combo_score = today_data["combo_score"]
exposure    = today_data["exposure"]
btc_price   = today_data["btc_price"]
counts      = today_data["signal_counts"]

# Signal label: combo < 0 means signals are more bearish (high → sell), but
# in this strategy negative combo → bullish (more signals pointing up)
signal_label = "BULLISH" if combo_score < 0 else "BEARISH" if combo_score > 0 else "NEUTRAL"
signal_color = "normal" if combo_score < 0 else "inverse" if combo_score > 0 else "off"

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("BTC Price", f"${btc_price:,.0f}")
with col2:
    st.metric("BTC Exposure", f"{exposure*100:.0f}%",
              delta=f"{(exposure - 0.5)*100:+.0f}% vs neutral")
with col3:
    st.metric("Signal", signal_label, delta=f"combo: {combo_score:.4f}",
              delta_color=signal_color)
with col4:
    st.metric("Bullish signals", counts["bullish"],
              delta=f"{counts['bullish']} / {sum(counts.values())}")
with col5:
    st.metric("D10 Sharpe", f"{metrics['d10']['sharpe']:.2f}",
              delta=f"HO: {metrics['d10']['ho_sharpe']:.2f}")

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Price & Exposure", "Signals Today", "Combo Score", "Performance"])

# ── TAB 1: BTC Price + Exposure overlay ──────────────────────────────────────
with tab1:
    st.subheader("BTC Price & Strategy Exposure")

    lookback = st.select_slider("Lookback", ["6M", "1Y", "2Y", "All"], value="2Y")
    days_map = {"6M": 180, "1Y": 365, "2Y": 730, "All": len(history)}
    df_plot = history.tail(days_map[lookback]).copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Exposure as filled area
    fig.add_trace(
        go.Scatter(
            x=df_plot.index, y=df_plot["exposure"] * 100,
            name="BTC Exposure %",
            fill="tozeroy",
            line=dict(color="rgba(99,110,250,0.8)", width=1),
            fillcolor="rgba(99,110,250,0.15)",
        ),
        secondary_y=True,
    )

    # BTC price (log scale)
    fig.add_trace(
        go.Scatter(
            x=df_plot.index, y=df_plot["btc_price"],
            name="BTC Price",
            line=dict(color="#F7931A", width=2),
        ),
        secondary_y=False,
    )

    fig.update_yaxes(title_text="BTC Price (USD)", type="log", secondary_y=False)
    fig.update_yaxes(title_text="Exposure %", range=[0, 110], secondary_y=True)
    fig.update_layout(
        height=500, hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, width='stretch')

# ── TAB 2: Signal breakdown ───────────────────────────────────────────────────
with tab2:
    st.subheader(f"Signal Breakdown — {today_data['date']}")

    signals = today_data["signals"]
    sig_df = pd.DataFrame([
        {"signal": k, "value": v["value"], "category": v["category"]}
        for k, v in signals.items()
    ]).sort_values(["category", "signal"])

    # Summary donut
    col_a, col_b = st.columns([1, 2])
    with col_a:
        bull = counts["bullish"]
        bear = counts["bearish"]
        neut = counts["neutral"]
        fig_pie = go.Figure(go.Pie(
            labels=["Bullish", "Bearish", "Neutral"],
            values=[bull, bear, neut],
            marker_colors=["#00cc96", "#ef553b", "#636efa"],
            hole=0.55,
            textinfo="label+value",
        ))
        fig_pie.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            annotations=[dict(text=signal_label, x=0.5, y=0.5,
                              font_size=16, showarrow=False)]
        )
        st.plotly_chart(fig_pie, width='stretch')

    with col_b:
        # Signal counts per category
        cat_summary = sig_df.groupby(["category", "value"]).size().unstack(fill_value=0)
        cat_summary = cat_summary.rename(columns={-1: "Bullish", 0: "Neutral", 1: "Bearish"})
        st.dataframe(
            cat_summary.style.background_gradient(cmap="RdYlGn", axis=None),
            width='stretch'
        )

    st.divider()

    # Full signal bar chart per category
    for cat in ["On-chain", "Technical", "Macro"]:
        cat_sigs = sig_df[sig_df["category"] == cat].copy()
        if cat_sigs.empty:
            continue
        with st.expander(f"{cat} signals ({len(cat_sigs)})", expanded=(cat == "On-chain")):
            color_map = {-1: "#00cc96", 0: "#aaaaaa", 1: "#ef553b"}
            colors = cat_sigs["value"].map(color_map).tolist()
            fig_bar = go.Figure(go.Bar(
                x=cat_sigs["value"],
                y=cat_sigs["signal"],
                orientation="h",
                marker_color=colors,
                text=cat_sigs["value"].map({-1: "Bullish", 0: "Neutral", 1: "Bearish"}),
                textposition="outside",
            ))
            fig_bar.update_layout(
                height=max(200, len(cat_sigs) * 22),
                xaxis=dict(range=[-1.5, 1.5], tickvals=[-1, 0, 1],
                           ticktext=["Bullish", "Neutral", "Bearish"]),
                margin=dict(l=0, r=60, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, width='stretch')

# ── TAB 3: Combo score history ────────────────────────────────────────────────
with tab3:
    st.subheader("Ensemble Combo Score History")
    st.caption("Negative combo = more bearish signals = strategy goes bullish (higher BTC exposure)")

    lookback2 = st.select_slider("Lookback ", ["6M", "1Y", "2Y", "All"], value="1Y")
    df_c = history.tail(days_map[lookback2]).copy()

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.6, 0.4], vertical_spacing=0.05)

    # Combo score with zero line
    fig2.add_hline(y=0, line_dash="dash", line_color="grey", row=1, col=1)
    fig2.add_trace(
        go.Scatter(
            x=df_c.index, y=df_c["combo"],
            name="Combo Score",
            line=dict(color="#636efa", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(99,110,250,0.12)",
        ),
        row=1, col=1,
    )

    # Exposure
    fig2.add_trace(
        go.Scatter(
            x=df_c.index, y=df_c["exposure"] * 100,
            name="Exposure %",
            line=dict(color="#F7931A", width=2),
        ),
        row=2, col=1,
    )
    fig2.add_hline(y=50, line_dash="dot", line_color="grey", row=2, col=1)

    fig2.update_yaxes(title_text="Combo Score", row=1, col=1)
    fig2.update_yaxes(title_text="Exposure %", range=[0, 105], row=2, col=1)
    fig2.update_layout(height=480, hovermode="x unified",
                       margin=dict(l=0, r=0, t=20, b=0),
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig2, width='stretch')

# ── TAB 4: Performance ────────────────────────────────────────────────────────
with tab4:
    st.subheader("Backtest Performance")

    d10 = metrics["d10"]
    bh  = metrics["bh"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**D10 Strategy**")
        st.metric("Annual Return",  f"{d10['ann_ret']:+.1%}")
        st.metric("Sharpe Ratio",   f"{d10['sharpe']:.2f}")
        st.metric("Max Drawdown",   f"{d10['max_dd']:.1%}")
        st.metric("Ann Volatility", f"{d10['ann_vol']:.1%}")
        st.metric("HO Sharpe",      f"{d10['ho_sharpe']:.2f}")
        st.metric("Avg Exposure",   f"{d10['avg_exposure']:.1%}")
        st.metric("Fees (bp/yr)",   f"{d10['fees']:.1f}")
    with c2:
        st.markdown("**BTC Buy & Hold**")
        st.metric("Annual Return",  f"{bh['ann_ret']:+.1%}")
        st.metric("Sharpe Ratio",   f"{bh['sharpe']:.2f}")
        st.metric("Max Drawdown",   f"{bh['max_dd']:.1%}")
        st.metric("Ann Volatility", f"{bh['ann_vol']:.1%}")
        st.metric("HO Sharpe",      f"{bh['ho_sharpe']:.2f}")

    st.divider()
    st.subheader("Yearly Returns")

    years = sorted(set(list(d10["yearly"].keys()) + list(bh["yearly"].keys())))
    yearly_df = pd.DataFrame({
        "Year":         years,
        "D10 Strategy": [d10["yearly"].get(y, float("nan")) for y in years],
        "BTC B&H":      [bh["yearly"].get(y,  float("nan")) for y in years],
    }).set_index("Year")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=years, y=[d10["yearly"].get(y, 0) * 100 for y in years],
        name="D10 Strategy", marker_color="#636efa",
    ))
    fig3.add_trace(go.Bar(
        x=years, y=[bh["yearly"].get(y, 0) * 100 for y in years],
        name="BTC B&H", marker_color="#F7931A", opacity=0.7,
    ))
    fig3.update_layout(
        barmode="group", height=350,
        yaxis_title="Return %",
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig3, width='stretch')

    # Styled table
    def _fmt(v):
        if pd.isna(v):
            return "-"
        color = "green" if v > 0 else "red"
        return f":{color}[{v:+.1%}]"

    styled = yearly_df.copy()
    for col in styled.columns:
        styled[col] = styled[col].apply(_fmt)
    st.dataframe(yearly_df.style.format("{:+.1%}").background_gradient(
        cmap="RdYlGn", vmin=-0.7, vmax=2.0
    ), width='stretch')

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("D10 Strategy v3 · Data refreshed daily via GitHub Actions · Not financial advice")