"""
app.py - AI Swing Trader Pro — Main Streamlit Dashboard
Indian Stock Market | NSE | Professional Swing Trading Platform
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Internal modules
from database import (
    init_database, get_watchlist, add_to_watchlist, remove_from_watchlist,
    save_scan_results, get_latest_scan, get_backtest_history, save_backtest
)
from data_fetcher import fetch_ohlcv, fetch_sector_indices
from indicators import calculate_all_indicators, get_latest_values
from scanner import run_scanner, run_watchlist_scan, get_scan_summary
from strategies import STRATEGIES
from ai_model import train_model, is_model_trained, get_feature_importance
from ranking_engine import get_score_label, create_score_breakdown_text
from institutional_tracker import get_fii_dii_data, get_institutional_sentiment, get_combined_flow_df, to_dataframe
from options_analyzer import get_option_analysis, format_oi_table
from alerts import (
    get_active_alerts, add_price_alert, add_score_alert,
    add_breakout_alert, check_alerts, test_telegram_connection
)
from backtester import run_backtest
from utils import (get_nse_symbols, get_bse_symbols, get_symbols, get_universe_options,
                   get_exchange_from_symbol, symbol_to_display,
                   EXCHANGE_NSE, EXCHANGE_BSE, EXCHANGE_BOTH,
                   NSE_SUFFIX, BSE_SUFFIX,
                   format_currency, format_number)

logger = logging.getLogger(__name__)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Swing Trader Pro | NSE & BSE India",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark professional theme */
:root {
    --bg-dark: #0d1117;
    --bg-card: #161b22;
    --accent-green: #00C853;
    --accent-red: #F44336;
    --accent-yellow: #FFD740;
    --accent-blue: #2979FF;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --border-color: #30363d;
}
.main { background-color: var(--bg-dark); }
.stButton>button {
    background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
    color: white; border-radius: 8px; border: none;
    padding: 0.4rem 1.2rem; font-weight: 600;
    transition: all 0.2s; letter-spacing: 0.5px;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(41,53,147,0.4); }
.metric-card {
    background: #161b22; border-radius: 12px; padding: 16px;
    border: 1px solid #30363d; text-align: center;
}
.score-pill {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-weight: 700; font-size: 0.9em;
}
.score-high { background: rgba(0,200,83,0.2); color: #00C853; }
.score-mid { background: rgba(255,215,64,0.2); color: #FFD740; }
.score-low { background: rgba(244,67,54,0.2); color: #F44336; }
.stDataFrame { border-radius: 10px; overflow: hidden; }
div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; }
.header-banner {
    background: linear-gradient(135deg, #0d1117 0%, #1a237e 50%, #0d1117 100%);
    border-radius: 12px; padding: 20px 30px; margin-bottom: 20px;
    border: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────────────────────
if "scan_results" not in st.session_state:
    st.session_state.scan_results = []
if "scan_running" not in st.session_state:
    st.session_state.scan_running = False
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "fii_data" not in st.session_state:
    st.session_state.fii_data = None
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None

init_database()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 AI Swing Trader Pro")
    st.markdown("*NSE & BSE India | Professional Platform*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Scanner", "📊 Stock Analysis",
         "🏦 Institutional Flow", "⚙️ Options Chain",
         "🤖 AI Model", "📋 Backtester",
         "👁️ Watchlist", "🔔 Alerts", "⚙️ Settings", "❓ Help"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### Scanner Settings")

    # Exchange selector
    exchange = st.selectbox("Exchange", [EXCHANGE_NSE, EXCHANGE_BSE, EXCHANGE_BOTH],
                             help="NSE (.NS), BSE (.BO), or scan both exchanges")

    # Universe dynamically updates based on exchange
    universe_opts = get_universe_options(exchange)
    universe = st.selectbox("Stock Universe", universe_opts)

    min_score = st.slider("Min AI Score", 0, 100, 50, 5)
    strategy_filter = st.multiselect(
        "Strategy Filter",
        list(STRATEGIES.values()),
        default=[]
    )

    st.divider()
    # Quick scan button in sidebar
    if st.button("⚡ Quick Scan", use_container_width=True):
        st.session_state.trigger_scan = True

    st.markdown(f"*Last scan: {st.session_state.last_scan_time or 'Never'}*")


# ─── Helper: Build Candlestick Chart ──────────────────────────────────────────

def build_stock_chart(symbol: str, df: pd.DataFrame = None) -> go.Figure:
    """Build interactive candlestick chart with indicators + subplots."""
    if df is None:
        with st.spinner(f"Fetching data for {symbol}..."):
            raw = fetch_ohlcv(symbol, period="6mo")
            if raw is None or raw.empty:
                return None
            df = calculate_all_indicators(raw)

    if df is None or df.empty:
        return None

    df = df.iloc[-120:]  # Last 120 bars

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=["Price", "Volume", "RSI", "MACD"]
    )

    display_sym = symbol.replace(".NS", "")

    # ── Candlestick ────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=display_sym,
        increasing_line_color="#00C853",
        decreasing_line_color="#F44336",
    ), row=1, col=1)

    # EMAs
    colors = {"EMA20": "#2979FF", "EMA50": "#FFD740", "EMA200": "#FF6D00"}
    for ema, color in colors.items():
        if ema in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ema], name=ema,
                line=dict(color=color, width=1.5, dash="solid"),
                opacity=0.85,
            ), row=1, col=1)

    # Bollinger Bands
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(128,128,255,0.5)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="rgba(128,128,255,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(128,128,255,0.04)",
        ), row=1, col=1)

    # Supertrend
    if "Supertrend" in df.columns and "Supertrend_Dir" in df.columns:
        bull_mask = df["Supertrend_Dir"] == 1
        bear_mask = df["Supertrend_Dir"] == -1
        fig.add_trace(go.Scatter(
            x=df.index[bull_mask], y=df["Supertrend"][bull_mask],
            name="Supertrend↑", mode="markers",
            marker=dict(color="#00C853", size=3, symbol="circle"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index[bear_mask], y=df["Supertrend"][bear_mask],
            name="Supertrend↓", mode="markers",
            marker=dict(color="#F44336", size=3, symbol="circle"),
        ), row=1, col=1)

    # ── Volume ─────────────────────────────────────────────────────────────────
    vol_colors = ["#00C853" if c >= o else "#F44336"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_colors, opacity=0.7,
    ), row=2, col=1)

    if "Vol_SMA20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Vol_SMA20"], name="Vol MA20",
            line=dict(color="#FFD740", width=1.2),
        ), row=2, col=1)

    # ── RSI ────────────────────────────────────────────────────────────────────
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#B39DDB", width=1.5),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#F44336", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00C853", line_width=1, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#555", line_width=0.8, row=3, col=1)

    # ── MACD ───────────────────────────────────────────────────────────────────
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#2979FF", width=1.5),
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#FF6D00", width=1.5),
        ), row=4, col=1)
        if "MACD_Hist" in df.columns:
            hist_colors = ["#00C853" if v >= 0 else "#F44336" for v in df["MACD_Hist"]]
            fig.add_trace(go.Bar(
                x=df.index, y=df["MACD_Hist"], name="MACD Hist",
                marker_color=hist_colors, opacity=0.7,
            ), row=4, col=1)

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=750,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter, sans-serif", color="#e6edf3", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(22,27,34,0.8)",
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=20, t=40, b=20),
        title=dict(
            text=f"<b>{display_sym}</b> — Swing Trade Analysis",
            font=dict(size=16, color="#e6edf3"),
        ),
    )
    fig.update_yaxes(gridcolor="#21262d", gridwidth=0.5)
    fig.update_xaxes(gridcolor="#21262d", gridwidth=0.5, showspikes=True)

    return fig


# ─── Page: Dashboard ──────────────────────────────────────────────────────────

def page_dashboard():
    st.markdown("""
    <div class="header-banner">
        <h1 style="margin:0; color:#e6edf3; font-size:1.8rem;">
            📈 AI Swing Trader Pro
        </h1>
        <p style="margin:4px 0 0; color:#8b949e; font-size:0.95rem;">
            NSE India | Institutional + Technical + AI Analysis | Find 5-10% Moves in 5-15 Days
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Market Overview
    with st.spinner("Loading market data..."):
        sector_data = fetch_sector_indices()

    # Metrics row
    nifty = sector_data.get("NIFTY50", {})
    bank = sector_data.get("NIFTY Bank", {})
    it = sector_data.get("NIFTY IT", {})
    pharma = sector_data.get("NIFTY Pharma", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    def metric_delta(data, key):
        val = data.get("change_pct", 0)
        return f"{val:+.2f}%" if val else "—"

    with col1:
        st.metric("NIFTY 50", f"₹{nifty.get('price', 0):,.0f}", metric_delta(nifty, "change_pct"))
    with col2:
        st.metric("NIFTY Bank", f"₹{bank.get('price', 0):,.0f}", metric_delta(bank, "change_pct"))
    with col3:
        st.metric("NIFTY IT", f"₹{it.get('price', 0):,.0f}", metric_delta(it, "change_pct"))
    with col4:
        st.metric("NIFTY Pharma", f"₹{pharma.get('price', 0):,.0f}", metric_delta(pharma, "change_pct"))
    with col5:
        results = st.session_state.scan_results
        st.metric("Opportunities", f"{len([r for r in results if r.get('ai_score',0)>=65])}", f"of {len(results)} scanned")

    st.divider()

    # Sector performance heatmap
    if sector_data:
        st.subheader("📊 Sector Performance")
        sec_df = pd.DataFrame([
            {"Sector": name, "1D Change %": data.get("change_pct", 0), "1W Change %": data.get("week_change_pct", 0)}
            for name, data in sector_data.items()
        ]).sort_values("1D Change %", ascending=False)

        colors = ["#00C853" if v >= 0 else "#F44336" for v in sec_df["1D Change %"]]
        fig = go.Figure(go.Bar(
            x=sec_df["Sector"],
            y=sec_df["1D Change %"],
            marker_color=colors,
            text=[f"{v:+.2f}%" for v in sec_df["1D Change %"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=300, template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            margin=dict(l=20, r=20, t=20, b=80),
            showlegend=False,
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top opportunities from last scan
    results = st.session_state.scan_results
    if results:
        st.subheader("🔥 Top Swing Opportunities")
        top = [r for r in results if r.get("ai_score", 0) >= 60][:10]
        if top:
            _render_results_table(top, compact=True)
    else:
        st.info("📌 Run a scan from the **Scanner** page to see top opportunities here.")

    # FII/DII quick view
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏦 Institutional Flow (Latest)")
        fii_data = get_fii_dii_data()
        fii_df, dii_df = to_dataframe(fii_data)
        sentiment = get_institutional_sentiment(fii_data)
        color = "#00C853" if sentiment == "bullish" else ("#F44336" if sentiment == "bearish" else "#FFD740")
        st.markdown(f"**FII Sentiment:** <span style='color:{color}'>{sentiment.upper()}</span>", unsafe_allow_html=True)
        if not fii_df.empty:
            recent = fii_df.head(7)
            fig = go.Figure(go.Bar(
                x=recent["date"], y=recent["net"],
                marker_color=["#00C853" if v > 0 else "#F44336" for v in recent["net"]],
                text=[f"₹{v:+.0f}Cr" for v in recent["net"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=220, template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                margin=dict(l=10, r=10, t=10, b=50),
                title="FII Daily Net (₹ Cr)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💼 DII Flow (Latest)")
        if not dii_df.empty:
            recent = dii_df.head(7)
            fig = go.Figure(go.Bar(
                x=recent["date"], y=recent["net"],
                marker_color=["#00C853" if v > 0 else "#F44336" for v in recent["net"]],
                text=[f"₹{v:+.0f}Cr" for v in recent["net"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=220, template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                margin=dict(l=10, r=10, t=10, b=50),
                title="DII Daily Net (₹ Cr)",
            )
            st.plotly_chart(fig, use_container_width=True)


# ─── Page: Scanner ─────────────────────────────────────────────────────────────

def page_scanner():
    st.title("🔍 AI Stock Scanner")

    # Exchange badge
    exchange_colors = {EXCHANGE_NSE: "#2979FF", EXCHANGE_BSE: "#FF6D00", EXCHANGE_BOTH: "#00C853"}
    exc_color = exchange_colors.get(exchange, "#2979FF")
    st.markdown(
        f"Scanning: <span style='background:{exc_color}22; color:{exc_color}; "
        f"padding:3px 12px; border-radius:20px; font-weight:700'>"
        f"{exchange} — {universe}</span>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        run_btn = st.button("🚀 Run Full Scan", use_container_width=True, type="primary")
    with col2:
        wl_btn = st.button("⭐ Scan Watchlist", use_container_width=True)
    with col3:
        load_prev = st.button("📂 Load Last Scan", use_container_width=True)
    with col4:
        train_btn = st.button("🤖 Train AI Model", use_container_width=True)

    triggered = run_btn or st.session_state.get("trigger_scan", False)
    if "trigger_scan" in st.session_state:
        del st.session_state["trigger_scan"]

    if triggered or wl_btn:
        # Get FII sentiment
        fii_data = get_fii_dii_data()
        fii_sentiment = get_institutional_sentiment(fii_data)
        st.session_state.fii_data = fii_data

        if wl_btn:
            watchlist = get_watchlist()
            if not watchlist:
                st.warning("Watchlist is empty. Add stocks first!")
                return
            # Watchlist symbols may have .NS or .BO or neither — normalise
            symbols = []
            for w in watchlist:
                sym = w["symbol"]
                if ".NS" not in sym and ".BO" not in sym:
                    sym = f"{sym}.NS"  # default to NSE
                symbols.append(sym)
        else:
            symbols = get_symbols(universe, exchange)

        progress_bar = st.progress(0, text=f"Scanning {len(symbols)} stocks...")
        status_text = st.empty()

        def update_progress(pct):
            progress_bar.progress(min(pct, 100) / 100, text=f"Analyzing... {pct}%")
            status_text.text(f"Progress: {pct}% | Stocks analyzed: ~{int(pct * len(symbols) / 100)}")

        strat_filter_keys = [k for k, v in STRATEGIES.items() if v in strategy_filter]

        with st.spinner("Running AI Scan..."):
            results = run_scanner(
                symbols=symbols,
                fii_sentiment=fii_sentiment,
                min_score=min_score,
                strategies_filter=strat_filter_keys if strat_filter_keys else None,
                progress_callback=update_progress,
            )

        st.session_state.scan_results = results
        st.session_state.last_scan_time = datetime.now().strftime("%H:%M:%S")

        # Save to DB
        save_scan_results(results)

        progress_bar.progress(1.0, text="✅ Scan Complete!")
        st.success(f"✅ Found {len(results)} opportunities from {len(symbols)} stocks!")

    elif load_prev:
        df = get_latest_scan()
        if not df.empty:
            st.session_state.scan_results = df.to_dict("records")
            st.success(f"Loaded {len(df)} results from last scan.")

    elif train_btn:
        with st.spinner("Training AI model on Nifty 50 stocks..."):
            from data_fetcher import fetch_batch_ohlcv
            from indicators import calculate_all_indicators
            symbols = get_nse_symbols("NIFTY50")
            data = fetch_batch_ohlcv(symbols[:30], period="2y")
            indicator_data = {}
            for sym, df in data.items():
                if df is not None and not df.empty and len(df) >= 100:
                    indicator_data[sym] = calculate_all_indicators(df)

            metrics = train_model(indicator_data)
            st.success("✅ AI Model trained!")
            st.json(metrics)

    # Display results
    results = st.session_state.scan_results
    if results:
        summary = get_scan_summary(results)

        # Summary metrics
        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Found", summary.get("total_scanned", 0))
        c2.metric("🔥 Strong Buy (80+)", summary.get("strong_buys", 0))
        c3.metric("✅ Buy (65-80)", summary.get("buys", 0))
        c4.metric("⚡ Watch (50-65)", summary.get("watches", 0))
        c5.metric("Avg AI Score", summary.get("avg_score", 0))

        st.divider()
        st.subheader("📋 Scan Results")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox("Sort By", ["AI Score", "RSI", "Volume Spike", "Price", "Prob 5%"])
        with col2:
            search_term = st.text_input("🔎 Search Stock", "")
        with col3:
            show_top = st.selectbox("Show", [20, 50, 100, "All"])

        # Apply filters
        display_results = results
        if search_term:
            display_results = [r for r in display_results
                                if search_term.upper() in r.get("display_symbol", "").upper()]

        sort_map = {
            "AI Score": "ai_score", "RSI": "rsi",
            "Volume Spike": "volume_spike", "Price": "price", "Prob 5%": "prob_5pct"
        }
        sort_key = sort_map.get(sort_by, "ai_score")
        display_results = sorted(display_results, key=lambda x: x.get(sort_key, 0), reverse=True)

        if show_top != "All":
            display_results = display_results[:int(show_top)]

        _render_results_table(display_results)


def _render_results_table(results: list, compact: bool = False):
    """Render the scanner results as an interactive table."""
    if not results:
        st.info("No results to display.")
        return

    rows = []
    for r in results:
        label, _ = get_score_label(r.get("ai_score", 0))
        sym = r.get("symbol", "")
        exch = "BSE" if sym.endswith(".BO") else "NSE"
        rows.append({
            "Symbol": r.get("display_symbol", ""),
            "Exch": exch,
            "Price (₹)": f"₹{r.get('price', 0):,.2f}",
            "RSI": f"{r.get('rsi', 0):.1f}",
            "Vol Spike": f"{r.get('volume_spike', 0):.1f}x",
            "Strategy": r.get("strategy", "—")[:35],
            "AI Score": r.get("ai_score", 0),
            "Signal": label,
            "Prob 5%": f"{r.get('prob_5pct', 0)*100:.0f}%",
            "Prob 10%": f"{r.get('prob_10pct', 0)*100:.0f}%",
            "Entry": f"₹{r.get('entry', 0):,.2f}",
            "SL": f"₹{r.get('stop_loss', 0):,.2f}",
            "T1": f"₹{r.get('target1', 0):,.2f}",
            "R:R": f"{r.get('risk_reward', 0):.1f}x",
        })

    df = pd.DataFrame(rows)

    if compact:
        df = df[["Symbol", "Exch", "Price (₹)", "RSI", "AI Score", "Signal", "Strategy", "R:R"]]

    # Color-code the AI Score column
    def highlight_score(val):
        try:
            s = float(val)
            if s >= 80:
                return "background-color: rgba(0,200,83,0.2); color: #00C853; font-weight: bold"
            elif s >= 65:
                return "background-color: rgba(0,200,83,0.1); color: #69F0AE"
            elif s >= 50:
                return "background-color: rgba(255,215,64,0.1); color: #FFD740"
            else:
                return "background-color: rgba(244,67,54,0.1); color: #EF9A9A"
        except Exception:
            return ""

    styled = df.style.applymap(highlight_score, subset=["AI Score"])
    st.dataframe(styled, use_container_width=True, height=500)

    # Click to analyze
    st.markdown("**Click a stock to analyze:**")
    if results:
        # Build label "RELIANCE (NSE)" for display
        sym_labels = [
            f"{r.get('display_symbol','')} ({'BSE' if r.get('symbol','').endswith('.BO') else 'NSE'})"
            for r in results[:50]
        ]
        sym_map = {label: r.get("symbol", "") for label, r in zip(sym_labels, results[:50])}
        selected_label = st.selectbox("Select stock", sym_labels, label_visibility="collapsed")
        if st.button("📊 Analyze Selected Stock", use_container_width=False):
            st.session_state.selected_stock = sym_map.get(selected_label, "")
            st.session_state.nav_override = "📊 Stock Analysis"
            st.rerun()


# ─── Page: Stock Analysis ──────────────────────────────────────────────────────

def page_stock_analysis():
    st.title("📊 Stock Analysis")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        symbol_input = st.text_input(
            "Enter Symbol (e.g. RELIANCE, TCS, HDFCBANK)",
            value=symbol_to_display(st.session_state.get("selected_stock") or "RELIANCE.NS")
        ).upper()
    with col2:
        exch_input = st.selectbox("Exchange", [EXCHANGE_NSE, EXCHANGE_BSE],
                                   index=0, help="NSE = .NS suffix | BSE = .BO suffix")
    with col3:
        period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)

    analyze_btn = st.button("🔍 Analyze", type="primary")

    if analyze_btn or st.session_state.get("selected_stock"):
        if st.session_state.get("selected_stock"):
            symbol = st.session_state.selected_stock
            del st.session_state["selected_stock"]
        else:
            suffix = BSE_SUFFIX if exch_input == EXCHANGE_BSE else NSE_SUFFIX
            if symbol_input.endswith(".NS") or symbol_input.endswith(".BO"):
                symbol = symbol_input
            else:
                symbol = f"{symbol_input}{suffix}"

        display_sym = symbol_to_display(symbol)
        exch_badge = "BSE" if symbol.endswith(".BO") else "NSE"
        exch_badge_color = "#FF6D00" if exch_badge == "BSE" else "#2979FF"

        with st.spinner(f"Loading {symbol}..."):
            raw_df = fetch_ohlcv(symbol, period=period)
            if raw_df is None or raw_df.empty:
                st.error(f"No data found for {symbol}. Check the symbol and try again.")
                return

            df = calculate_all_indicators(raw_df)
            ind = get_latest_values(df)

        from strategies import check_all_strategies, generate_trade_plan
        from ai_model import predict_move_probability
        from ranking_engine import calculate_ai_score

        triggered = check_all_strategies(df, ind)
        ml_probs = predict_move_probability(df)
        score_dict = calculate_ai_score(ind, triggered, ml_probs)
        trade_plan = generate_trade_plan(df, ind, triggered[0]["key"] if triggered else "DEFAULT")

        # Header metrics
        price = ind.get("Close", 0)
        label, color = get_score_label(score_dict["total_score"])

        st.markdown(f"""
        <div style="background:#161b22; border-radius:12px; padding:20px; border:1px solid #30363d; margin-bottom:20px;">
            <h2 style="color:#e6edf3; margin:0">{display_sym}
              <span style="font-size:0.7rem; background:{exch_badge_color}22; color:{exch_badge_color};
                padding:2px 10px; border-radius:12px; margin-left:10px; vertical-align:middle;">{exch_badge}</span>
            </h2>
            <div style="display:flex; gap:30px; margin-top:10px; flex-wrap:wrap;">
                <div><span style="color:#8b949e">Price</span><br><b style="font-size:1.4rem; color:#e6edf3">₹{price:,.2f}</b></div>
                <div><span style="color:#8b949e">RSI</span><br><b style="font-size:1.4rem; color:#B39DDB">{ind.get('RSI',0):.1f}</b></div>
                <div><span style="color:#8b949e">ADX</span><br><b style="font-size:1.4rem; color:#FFD740">{ind.get('ADX',0):.1f}</b></div>
                <div><span style="color:#8b949e">Vol Spike</span><br><b style="font-size:1.4rem; color:#2979FF">{ind.get('Vol_Spike',0):.1f}x</b></div>
                <div><span style="color:#8b949e">AI Score</span><br><b style="font-size:1.4rem; color:{color}">{score_dict['total_score']:.0f}/100 {label}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chart
        fig = build_stock_chart(symbol, df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Two-column analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚡ Strategy Signals")
            if triggered:
                for s in triggered:
                    with st.expander(f"✅ {s['name']} — Strength: {s['strength']}/100", expanded=True):
                        st.markdown(s.get("description", ""))
                        details = s.get("details", {})
                        for k, v in details.items():
                            icon = "✅" if v else "❌"
                            st.markdown(f"{icon} `{k.replace('_', ' ').title()}`")
            else:
                st.info("No strategies triggered currently.")

            st.subheader("📈 AI Score Breakdown")
            breakdown = create_score_breakdown_text(score_dict)
            st.code(breakdown, language=None)

        with col2:
            st.subheader("📋 Trade Plan")
            if trade_plan:
                tp_col1, tp_col2 = st.columns(2)
                with tp_col1:
                    st.metric("Entry Price", f"₹{trade_plan.get('entry',0):,.2f}")
                    st.metric("Target 1 (2R)", f"₹{trade_plan.get('target1',0):,.2f}")
                    st.metric("Risk Amount", f"₹{trade_plan.get('risk_amount',0):,.2f}")
                with tp_col2:
                    st.metric("Stop Loss", f"₹{trade_plan.get('stop_loss',0):,.2f}")
                    st.metric("Target 2 (3R)", f"₹{trade_plan.get('target2',0):,.2f}")
                    st.metric("Risk:Reward", f"{trade_plan.get('risk_reward',0):.1f}x")

                st.metric("ATR (Volatility)", f"₹{trade_plan.get('atr',0):,.2f}")

                # Gauge chart for AI score
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_dict["total_score"],
                    title={"text": "AI Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2979FF"},
                        "steps": [
                            {"range": [0, 35], "color": "#1a1a2e"},
                            {"range": [35, 65], "color": "#1a2a3a"},
                            {"range": [65, 80], "color": "#1a3a2a"},
                            {"range": [80, 100], "color": "#1a4a2a"},
                        ],
                        "threshold": {
                            "line": {"color": "#00C853", "width": 3},
                            "thickness": 0.75,
                            "value": 80
                        },
                    },
                    number={"suffix": "/100", "font": {"color": "#e6edf3"}},
                ))
                fig_gauge.update_layout(
                    height=250, template="plotly_dark",
                    paper_bgcolor="#161b22", margin=dict(l=30, r=30, t=40, b=10),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.subheader("🤖 ML Predictions")
            prob_data = {
                "5% Move": ml_probs.get("prob_5pct", 0) * 100,
                "8% Move": ml_probs.get("prob_8pct", 0) * 100,
                "10% Move": ml_probs.get("prob_10pct", 0) * 100,
            }
            for label_p, prob in prob_data.items():
                col_l, col_b = st.columns([1, 2])
                col_l.write(label_p)
                col_b.progress(prob / 100, text=f"{prob:.0f}%")

            st.caption(f"Model: `{ml_probs.get('model_used', 'fallback')}`")

        # Add to watchlist
        st.divider()
        col_w1, col_w2 = st.columns([2, 1])
        with col_w1:
            if st.button(f"⭐ Add {symbol.replace('.NS','')} to Watchlist"):
                if add_to_watchlist(symbol.replace(".NS", "")):
                    st.success("Added to watchlist!")
                else:
                    st.info("Already in watchlist.")
        with col_w2:
            if st.button("🔔 Set Alert"):
                add_score_alert(symbol.replace(".NS", ""), 75)
                st.success("Alert set for AI Score ≥ 75!")

        # Indicator table
        with st.expander("📊 Full Indicator Values"):
            ind_df = pd.DataFrame([{
                "Indicator": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)
            } for k, v in ind.items() if not k.startswith("_")])
            st.dataframe(ind_df, use_container_width=True)


# ─── Page: Institutional Flow ─────────────────────────────────────────────────

def page_institutional():
    st.title("🏦 Institutional Flow Tracker")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Data"):
            st.session_state.fii_data = None

    with st.spinner("Fetching institutional data..."):
        fii_data = st.session_state.get("fii_data") or get_fii_dii_data(force_refresh=True)
        st.session_state.fii_data = fii_data

    fii_df, dii_df = to_dataframe(fii_data)
    sentiment = get_institutional_sentiment(fii_data)

    # Summary
    color_map = {"bullish": "#00C853", "bearish": "#F44336", "neutral": "#FFD740"}
    color = color_map.get(sentiment, "#FFD740")

    if not fii_df.empty:
        fii_5d = fii_df["net"].head(5).sum()
        fii_10d = fii_df["net"].head(10).sum()
        dii_5d = dii_df["net"].head(5).sum() if not dii_df.empty else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Market Sentiment", sentiment.upper(), delta_color="normal")
        col2.metric("FII 5-Day Net", f"₹{fii_5d:+,.0f}Cr",
                    delta="Buying" if fii_5d > 0 else "Selling")
        col3.metric("FII 10-Day Net", f"₹{fii_10d:+,.0f}Cr",
                    delta="Buying" if fii_10d > 0 else "Selling")
        col4.metric("DII 5-Day Net", f"₹{dii_5d:+,.0f}Cr",
                    delta="Buying" if dii_5d > 0 else "Selling")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["FII Flow", "DII Flow", "Combined Analysis"])

    with tab1:
        if not fii_df.empty:
            _render_flow_chart(fii_df, "FII (Foreign Institutional Investors)")
            st.dataframe(fii_df.head(20).style.applymap(
                lambda v: f"color: {'#00C853' if v > 0 else '#F44336'}" if isinstance(v, (int, float)) else "",
                subset=["net"]
            ), use_container_width=True)

    with tab2:
        if not dii_df.empty:
            _render_flow_chart(dii_df, "DII (Domestic Institutional Investors)")
            st.dataframe(dii_df.head(20), use_container_width=True)

    with tab3:
        combined = get_combined_flow_df(fii_data)
        if not combined.empty:
            fig = go.Figure()
            for col, color in [("FII_Net", "#2979FF"), ("DII_Net", "#00C853"), ("Total_Net", "#FFD740")]:
                if col in combined.columns:
                    fig.add_trace(go.Scatter(
                        x=combined["date"], y=combined[col],
                        name=col.replace("_", " "), mode="lines+markers",
                        line=dict(color=color, width=2),
                    ))
            fig.update_layout(
                height=400, template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                title="FII + DII Combined Flow (₹ Cr)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Institutional flags on current scan
        if st.session_state.scan_results:
            st.subheader("🏦 Institutional + Breakout Flags")
            from institutional_tracker import flag_institutional_stocks
            flagged = flag_institutional_stocks(st.session_state.scan_results, fii_data)
            flagged_stocks = [s for s in flagged if s.get("inst_flag")]
            if flagged_stocks:
                _render_results_table(flagged_stocks[:20])
            else:
                st.info("No stocks with combined institutional + breakout signal. Run a scan first.")


def _render_flow_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"].head(20), y=df["net"].head(20),
        marker_color=["#00C853" if v > 0 else "#F44336" for v in df["net"].head(20)],
        name="Net Flow",
        text=[f"₹{v:+,.0f}" for v in df["net"].head(20)],
        textposition="outside",
    ))
    if "buy" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"].head(20), y=df["buy"].head(20),
            name="Buy", mode="lines+markers",
            line=dict(color="#69F0AE", width=1.5, dash="dot"),
        ))
    fig.update_layout(
        height=380, template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        title=f"{title} — Net Flow (₹ Cr)",
        yaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Page: Options Chain ──────────────────────────────────────────────────────

def page_options():
    st.title("⚙️ Options Chain Analyzer")

    col1, col2 = st.columns([3, 1])
    with col1:
        oc_symbol = st.text_input("NSE Symbol", "NIFTY").upper()
    with col2:
        fetch_btn = st.button("🔍 Fetch Chain", type="primary")

    if fetch_btn:
        with st.spinner(f"Fetching option chain for {oc_symbol}..."):
            analysis = get_option_analysis(oc_symbol, force_refresh=True)

        price = analysis.get("current_price", 0)
        pcr = analysis.get("pcr", 0)
        max_pain = analysis.get("max_pain", 0)
        sentiment = analysis.get("sentiment", "neutral")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current Price", f"₹{price:,.2f}")
        col2.metric("Put/Call Ratio", f"{pcr:.2f}", delta="Bullish" if pcr > 1 else "Bearish")
        col3.metric("Max Pain", f"₹{max_pain:,.0f}")
        col4.metric("Total Call OI", format_number(analysis.get("call_oi_total", 0)))
        col5.metric("Total Put OI", format_number(analysis.get("put_oi_total", 0)))

        # Signal
        cond = analysis.get("market_condition", "neutral")
        bull_sig = analysis.get("bullish_signal", False)
        bear_sig = analysis.get("bearish_signal", False)

        if bull_sig:
            st.success(f"🐂 **Bullish Signal**: {cond.replace('_', ' ').title()} | Sentiment: {sentiment}")
        elif bear_sig:
            st.error(f"🐻 **Bearish Signal**: {cond.replace('_', ' ').title()} | Sentiment: {sentiment}")
        else:
            st.info(f"⚖️ **Neutral** | {cond.replace('_', ' ').title()} | Sentiment: {sentiment}")

        # OI Chart
        oc_df = format_oi_table(analysis)
        if not oc_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=oc_df["Strike"], y=oc_df["Call OI"],
                name="Call OI", marker_color="#F44336", opacity=0.8,
            ))
            fig.add_trace(go.Bar(
                x=oc_df["Strike"], y=oc_df["Put OI"],
                name="Put OI", marker_color="#00C853", opacity=0.8,
            ))
            if price > 0:
                fig.add_vline(x=price, line_dash="dash", line_color="#FFD740",
                              annotation_text=f"LTP: ₹{price:.0f}")
            if max_pain > 0:
                fig.add_vline(x=max_pain, line_dash="dot", line_color="#FF6D00",
                              annotation_text=f"Max Pain: ₹{max_pain:.0f}")
            fig.update_layout(
                height=450, barmode="group", template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                title=f"{oc_symbol} — Open Interest Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Support & Resistance
            col1, col2 = st.columns(2)
            with col1:
                resistance = analysis.get("resistance_levels", [])
                st.markdown("**🔴 Resistance Levels (High Call OI)**")
                for r in resistance[:3]:
                    st.markdown(f"• ₹{r:,.0f}")
            with col2:
                support = analysis.get("support_levels", [])
                st.markdown("**🟢 Support Levels (High Put OI)**")
                for s in support[:3]:
                    st.markdown(f"• ₹{s:,.0f}")

            st.subheader("📋 Option Chain Table")
            st.dataframe(oc_df, use_container_width=True, height=400)


# ─── Page: AI Model ───────────────────────────────────────────────────────────

def page_ai_model():
    st.title("🤖 AI Model Manager")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Status")
        if is_model_trained():
            st.success("✅ ML Model is trained and ready")
        else:
            st.warning("⚠️ Model not trained yet. Using rule-based predictions.")

        st.subheader("Train New Model")
        train_universe = st.selectbox("Training Universe", ["NIFTY50", "NIFTY100"])
        train_period = st.selectbox("Training Period", ["1y", "2y", "3y"], index=1)

        if st.button("🚀 Train Model", type="primary"):
            from data_fetcher import fetch_batch_ohlcv
            from indicators import calculate_all_indicators

            symbols = get_nse_symbols(train_universe)
            progress = st.progress(0)
            st.write(f"Fetching data for {len(symbols)} stocks...")

            data = fetch_batch_ohlcv(symbols, period=train_period)
            progress.progress(0.4)

            indicator_data = {}
            for sym, df in data.items():
                if df is not None and not df.empty and len(df) >= 100:
                    indicator_data[sym] = calculate_all_indicators(df)
            progress.progress(0.7)

            metrics = train_model(indicator_data)
            progress.progress(1.0)

            st.success(f"✅ Model trained on {len(indicator_data)} stocks!")
            st.json(metrics)

    with col2:
        st.subheader("Feature Importance")
        importance = get_feature_importance()
        if importance:
            imp_df = pd.DataFrame([
                {"Feature": k, "Importance": v}
                for k, v in list(importance.items())[:15]
            ])
            fig = go.Figure(go.Bar(
                x=imp_df["Importance"], y=imp_df["Feature"],
                orientation="h",
                marker_color="#2979FF",
            ))
            fig.update_layout(
                height=450, template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                title="Top 15 Feature Importances",
                margin=dict(l=150, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train the model to see feature importances.")

        st.subheader("Model Architecture")
        st.markdown("""
        **Algorithm:** Random Forest Classifier
        - 3 binary classifiers: 5%, 8%, 10% move
        - 100 decision trees per model
        - Features: 23 technical indicators
        - Forward window: 10 trading days
        - Training: 80% / Testing: 20% split
        - Class balancing: Enabled
        """)


# ─── Page: Backtester ─────────────────────────────────────────────────────────

def page_backtester():
    st.title("📋 Strategy Backtester")

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_symbol = st.text_input("Symbol", "RELIANCE").upper()
    with col2:
        bt_strategy = st.selectbox("Strategy", ["ALL"] + list(STRATEGIES.keys()),
                                    format_func=lambda x: x if x == "ALL" else STRATEGIES.get(x, x))
    with col3:
        bt_capital = st.number_input("Initial Capital (₹)", value=100000, step=10000)

    col1, col2 = st.columns(2)
    with col1:
        bt_start = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        bt_end = st.date_input("End Date", value=datetime.now())

    if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
        symbol = f"{bt_symbol}.NS"
        with st.spinner(f"Backtesting {bt_symbol} — {bt_strategy}..."):
            result = run_backtest(
                symbol=symbol,
                strategy_key=bt_strategy,
                start_date=str(bt_start),
                end_date=str(bt_end),
                initial_capital=bt_capital,
            )

        if "error" in result:
            st.error(result["error"])
            return

        metrics = result["metrics"]
        trades = result["trades"]
        equity_curve = result["equity_curve"]

        # Metrics
        st.divider()
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Trades", metrics["total_trades"])
        col2.metric("Win Rate", f"{metrics['win_rate']:.1f}%",
                    delta="Good" if metrics["win_rate"] >= 50 else "Low",
                    delta_color="normal" if metrics["win_rate"] >= 50 else "inverse")
        col3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}",
                    delta="Positive" if metrics["profit_factor"] > 1 else "Negative")
        col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
        col5.metric("Total Return", f"{metrics['total_return_pct']:.1f}%",
                    delta_color="normal" if metrics["total_return_pct"] >= 0 else "inverse")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Return/Trade", f"{metrics['avg_return']:.2f}%")
        col2.metric("Avg Win", f"{metrics.get('avg_win', 0):.2f}%")
        col3.metric("Avg Loss", f"{metrics.get('avg_loss', 0):.2f}%")

        # Equity curve
        st.subheader("📈 Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=equity_curve, mode="lines",
            line=dict(color="#00C853" if equity_curve[-1] >= bt_capital else "#F44336", width=2),
            fill="tozeroy", fillcolor="rgba(0,200,83,0.05)",
            name="Portfolio Value",
        ))
        fig.add_hline(y=bt_capital, line_dash="dash", line_color="#555",
                      annotation_text=f"Initial: ₹{bt_capital:,.0f}")
        fig.update_layout(
            height=350, template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            yaxis=dict(tickprefix="₹", gridcolor="#21262d"),
            title=f"{bt_symbol} — {bt_strategy} Equity Curve",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trades table
        if trades:
            st.subheader("📋 Trade Log")
            trades_df = pd.DataFrame(trades)

            def color_result(val):
                return "color: #00C853; font-weight: bold" if val == "WIN" else "color: #F44336"

            st.dataframe(
                trades_df.style.applymap(color_result, subset=["result"]),
                use_container_width=True, height=400
            )

        # Save result
        save_backtest(symbol, bt_strategy, str(bt_start), str(bt_end), metrics, trades)

    # Backtest history
    st.divider()
    st.subheader("📚 Backtest History")
    history = get_backtest_history()
    if not history.empty:
        st.dataframe(history[["run_date", "symbol", "strategy", "win_rate",
                               "avg_return", "profit_factor", "max_drawdown", "total_trades"]],
                     use_container_width=True)


# ─── Page: Watchlist ──────────────────────────────────────────────────────────

def page_watchlist():
    st.title("👁️ Watchlist")

    col1, col2 = st.columns([3, 1])
    with col1:
        new_sym = st.text_input("Add Symbol (e.g. INFY, TCS)", key="wl_input").upper()
    with col2:
        if st.button("➕ Add", use_container_width=True):
            if new_sym:
                if add_to_watchlist(new_sym):
                    st.success(f"Added {new_sym}!")
                else:
                    st.info(f"{new_sym} already in watchlist.")

    watchlist = get_watchlist()
    if watchlist:
        wl_df = pd.DataFrame(watchlist)
        st.dataframe(wl_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Scan Watchlist", use_container_width=True, type="primary"):
                fii_sentiment = get_institutional_sentiment()
                with st.spinner("Scanning watchlist..."):
                    results = run_watchlist_scan(watchlist, fii_sentiment)
                st.session_state.scan_results = results
                st.success(f"Scan complete! {len(results)} results.")
                _render_results_table(results)

        with col2:
            remove_sym = st.selectbox("Remove Symbol", [w["symbol"] for w in watchlist])
            if st.button("🗑️ Remove", use_container_width=True):
                remove_from_watchlist(remove_sym)
                st.success(f"Removed {remove_sym}")
                st.rerun()
    else:
        st.info("Your watchlist is empty. Add some stocks above!")


# ─── Page: Alerts ─────────────────────────────────────────────────────────────

def page_alerts():
    st.title("🔔 Alert Manager")

    with st.expander("➕ Create New Alert", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            alert_sym = st.text_input("Symbol", "RELIANCE").upper()
        with col2:
            alert_type = st.selectbox("Alert Type",
                                       ["Price Above", "Price Below", "AI Score ≥ 80",
                                        "Breakout", "Strategy Trigger"])
        with col3:
            alert_val = st.number_input("Target Value (for price/score alerts)", value=0.0)

        if st.button("🔔 Create Alert", type="primary"):
            if "Price Above" in alert_type:
                add_price_alert(alert_sym, alert_val, "above")
            elif "Price Below" in alert_type:
                add_price_alert(alert_sym, alert_val, "below")
            elif "AI Score" in alert_type:
                add_score_alert(alert_sym, 80)
            elif "Breakout" in alert_type:
                add_breakout_alert(alert_sym)
            else:
                from alerts import add_strategy_alert
                add_strategy_alert(alert_sym)
            st.success(f"Alert created for {alert_sym}!")

    # Active alerts
    st.subheader("🔔 Active Alerts")
    active = get_active_alerts()
    if active:
        df = pd.DataFrame(active)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No active alerts. Create one above!")

    # Telegram setup
    st.divider()
    st.subheader("📱 Telegram Notifications")
    with st.expander("Configure Telegram Bot"):
        bot_token = st.text_input("Bot Token", type="password",
                                   help="Get from @BotFather on Telegram")
        chat_id = st.text_input("Chat ID", help="Your Telegram Chat ID")
        if st.button("Test Connection"):
            import os
            os.environ["TELEGRAM_BOT_TOKEN"] = bot_token
            os.environ["TELEGRAM_CHAT_ID"] = chat_id
            if test_telegram_connection():
                st.success("✅ Telegram connected!")
            else:
                st.error("❌ Connection failed. Check token.")
        st.info("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as environment variables to persist.")


# ─── Page: Settings ───────────────────────────────────────────────────────────

def page_settings():
    st.title("⚙️ Settings")

    st.subheader("Scanner Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Min Volume Spike Filter", value=1.0, step=0.1)
        st.number_input("Min ADX for Trend Strength", value=20, step=5)
        st.slider("RSI Range for Momentum Breakout", 40, 80, (55, 75))
    with col2:
        st.number_input("Stop Loss Multiplier (× ATR)", value=1.5, step=0.1)
        st.number_input("Target R Multiple", value=2.0, step=0.5)
        st.number_input("Cache TTL (seconds)", value=300, step=60)

    st.divider()
    st.subheader("About")
    st.markdown("""
    **AI Swing Trader Pro** — Professional NSE India Swing Trading Platform

    **Features:**
    - 🔍 Scan NIFTY50/100/500/All NSE stocks
    - 🤖 AI/ML-powered scoring (Random Forest)
    - 📊 14+ technical indicators
    - 🏦 FII/DII institutional flow tracking
    - ⚙️ Options chain analysis (PCR, Max Pain, OI)
    - 📋 Strategy backtesting engine
    - 🔔 Telegram + desktop alerts
    - 📈 Interactive Plotly charts

    **Tech Stack:** Python, Streamlit, Plotly, yfinance, scikit-learn, SQLite
    """)


# ─── Page: Help ───────────────────────────────────────────────────────────────

def page_help():
    st.markdown("""
    <div class="header-banner">
        <h1 style="margin:0; color:#e6edf3; font-size:1.8rem;">❓ Help & User Guide</h1>
        <p style="margin:4px 0 0; color:#8b949e">
            Complete guide to using AI Swing Trader Pro — NSE & BSE India
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚀 Getting Started",
        "🔍 Scanner",
        "📊 Stock Analysis",
        "🤖 AI & Scoring",
        "📋 Strategies",
        "🔔 Alerts & BSE/NSE",
    ])

    with tab1:
        st.subheader("🚀 Getting Started — First Steps")
        st.markdown("""
        Welcome to **AI Swing Trader Pro**, a fully local, AI-powered platform for finding
        high-probability swing trades on Indian markets (NSE + BSE).

        ---

        ### Step-by-Step First Use

        **Step 1 — Run your first scan**
        1. In the **sidebar**, choose your **Exchange** (NSE, BSE, or Both).
        2. Choose a **Stock Universe** — start with **NIFTY50** for speed.
        3. Leave **Min AI Score** at 50 to see a reasonable number of results.
        4. Click **⚡ Quick Scan** in the sidebar or navigate to **🔍 Scanner** and click **🚀 Run Full Scan**.
        5. Wait for the progress bar to complete.

        **Step 2 — Review the results**
        - The scanner table shows every stock ranked by **AI Score** (0–100).
        - Stocks scoring **80+** are 🔥 Strong Buy candidates.
        - Click any stock and hit **📊 Analyze Selected Stock** to deep-dive.

        **Step 3 — Analyze a trade**
        - In **📊 Stock Analysis**, review the candlestick chart, indicators, and trade plan.
        - Check the **Entry / Stop Loss / Target** levels auto-generated for you.
        - Check the **ML Prediction** — probability of 5%/8%/10% move in 10 days.

        **Step 4 — Build your watchlist**
        - Navigate to **👁️ Watchlist**, add symbols you want to track daily.
        - Use **⭐ Scan Watchlist** in the Scanner to get updates on just those stocks.

        **Step 5 — Train the AI (optional but recommended)**
        - Go to **🤖 AI Model** and click **🚀 Train Model**.
        - This trains a Random Forest on 2 years of historical data for more accurate predictions.
        - Without training, rule-based predictions are used (still useful, but less precise).

        ---

        ### System Requirements
        | Item | Minimum |
        |------|---------|
        | Python | 3.9+ |
        | RAM | 4 GB |
        | Disk | 500 MB |
        | Internet | Required for data |

        ### Installation
        ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```
        """)

    with tab2:
        st.subheader("🔍 Scanner — How It Works")
        st.markdown("""
        The scanner is the heart of the platform. It analyses hundreds of stocks in parallel
        and surfaces the best swing trade setups.

        ---

        ### Exchange & Universe Selection (Sidebar)

        | Exchange | Universes Available | Suffix Used |
        |----------|--------------------|----|
        | **NSE** | NIFTY50, NIFTY100, NIFTY500, ALL NSE | `.NS` |
        | **BSE** | SENSEX30, BSE100, BSE_MIDCAP, BSE_ONLY, ALL BSE | `.BO` |
        | **Both** | NIFTY50+SENSEX30, NIFTY100+BSE100, etc. | Both |

        > **Tip:** Start with **NIFTY50** or **SENSEX30**. Larger universes take longer but find more opportunities.

        > **BSE_ONLY** contains stocks that trade exclusively or primarily on BSE and may not appear in NSE scans.

        ---

        ### Scanner Settings

        | Setting | What It Does |
        |---------|-------------|
        | **Min AI Score** | Only show stocks scoring above this threshold. 50 is balanced; 70+ for high-conviction only. |
        | **Strategy Filter** | Show only stocks matching specific strategies. Leave blank for all. |

        ---

        ### Reading the Results Table

        | Column | Meaning |
        |--------|---------|
        | **Symbol** | Stock ticker |
        | **Exch** | NSE or BSE |
        | **Price** | Current market price (₹) |
        | **RSI** | Relative Strength Index (14-period). 55–70 is ideal for momentum trades. |
        | **Vol Spike** | Current volume ÷ 20-day avg. >1.5x means unusual activity. |
        | **Strategy** | Which strategy triggered (can be multiple). |
        | **AI Score** | Overall score 0–100. Higher = better setup. |
        | **Signal** | 🔥 Strong Buy / ✅ Buy / ⚡ Watch / ⚠️ Weak |
        | **Prob 5%** | ML model probability of 5%+ move in next 10 days. |
        | **Entry / SL / T1** | Suggested entry, stop loss, and first target prices. |
        | **R:R** | Risk-to-reward ratio. >2x is good; >3x is excellent. |

        ---

        ### Performance Guide

        | Universe | Approx Time |
        |----------|------------|
        | NIFTY50 / SENSEX30 | 10–20 seconds |
        | NIFTY100 / BSE100 | 25–40 seconds |
        | NIFTY500 / BSE_MIDCAP | 2–4 minutes |
        | ALL NSE + ALL BSE | 15–25 minutes |

        ---

        ### Load Last Scan
        Saves you time — reloads the previous scan result from the local SQLite database
        without making new network requests.
        """)

    with tab3:
        st.subheader("📊 Stock Analysis — Deep Dive Guide")
        st.markdown("""
        The Stock Analysis page gives you a complete technical picture of any single stock.

        ---

        ### Entering a Symbol
        - Type the **base symbol** (e.g. `RELIANCE`, `INFY`, `HDFCBANK`).
        - Select **NSE** (for `.NS`) or **BSE** (for `.BO`) from the Exchange dropdown.
        - You can also enter the full symbol directly: `RELIANCE.NS` or `RELIANCE.BO`.

        ---

        ### Chart Explained

        The interactive Plotly chart has **4 panels**:

        | Panel | Contents |
        |-------|---------|
        | **Price** | Candlesticks + EMA20 (blue) + EMA50 (yellow) + EMA200 (orange) + Bollinger Bands + Supertrend dots |
        | **Volume** | Bar chart coloured green/red + 20-day volume moving average |
        | **RSI** | 14-period RSI with overbought (70) and oversold (30) lines |
        | **MACD** | MACD line, Signal line, and histogram |

        **Chart tips:**
        - Zoom: scroll wheel or drag to select a range
        - Pan: click and drag
        - Toggle indicators: click the legend items
        - Hover over any candle to see OHLCV details

        ---

        ### Trade Plan Interpretation

        | Field | How It's Calculated |
        |-------|---------------------|
        | **Entry** | Just above resistance (Breakout) or at current price (Pullback/Reversal) |
        | **Stop Loss** | Entry − 1.0–1.5 × ATR (depends on strategy) |
        | **Target 1** | Entry + 2 × Risk (2R) |
        | **Target 2** | Entry + 3 × Risk (3R) |
        | **Risk:Reward** | (Target1 − Entry) ÷ (Entry − Stop Loss) |

        > **Rule of thumb:** Only take trades where R:R ≥ 2.0.

        ---

        ### ML Prediction Bars
        Three progress bars show the probability of the stock moving up by 5%, 8%, or 10%
        within the **next 10 trading days**, based on the trained Random Forest model.

        | Probability | Interpretation |
        |-------------|---------------|
        | < 30% | Low conviction |
        | 30–50% | Moderate setup |
        | 50–70% | Good probability |
        | > 70% | High-conviction trade |
        """)

    with tab4:
        st.subheader("🤖 AI Score & ML Model")
        st.markdown("""
        ### AI Score (0–100)

        The AI Score is a **composite weighted score** that evaluates every stock on 5 dimensions:

        | Component | Weight | What It Measures |
        |-----------|--------|-----------------|
        | **Trend Strength** | 25% | EMA alignment (EMA20 > EMA50 > EMA200), ADX, Supertrend direction |
        | **Momentum** | 25% | RSI position (55–70 ideal), MACD crossover, Stochastic RSI |
        | **Volume Expansion** | 20% | Current volume vs 20-day average. 2x spike = 80/100 on this component |
        | **Breakout Quality** | 15% | 20-day high breakout, BB squeeze before breakout, volume confirmation |
        | **Institutional** | 15% | FII net buying/selling sentiment over last 5 trading days |

        **Score Thresholds:**
        - 🔥 **80–100** → Strong Buy — multiple confirmations, high probability setup
        - ✅ **65–79** → Buy — good setup, most conditions met
        - ⚡ **50–64** → Watch — developing setup, not all conditions confirmed
        - ⚠️ **35–49** → Weak — marginal setup, avoid or wait
        - ❌ **0–34** → Avoid — very few conditions met

        ---

        ### ML Model (Random Forest)

        **What it learns:**
        - Trained on 2+ years of historical OHLCV + indicator data
        - 3 separate classifiers: predicts probability of 5%, 8%, 10% upward move
        - Forward window: next 10 trading days
        - Features used: RSI, MACD, ADX, EMA alignment, Volume Spike, ATR%, BB%, StochRSI, price returns (1d/3d/5d/10d), and more

        **Training:**
        1. Go to **🤖 AI Model** page
        2. Select universe (NIFTY50 = fastest, NIFTY100 = more accurate)
        3. Select period (2y recommended for sufficient data)
        4. Click **🚀 Train Model**

        **Without a trained model**, the platform uses a rule-based fallback that still
        gives meaningful predictions based on indicator combinations.

        ---

        ### Feature Importance
        After training, the **Feature Importance** chart shows which indicators matter most
        to the model. This can help you understand what's driving high-scoring stocks.
        """)

    with tab5:
        st.subheader("📋 Swing Trading Strategies")
        st.markdown("""
        The platform implements **4 core swing trading strategies**. A stock can trigger
        multiple strategies simultaneously (which increases its AI score).

        ---

        ### Strategy 1 — 🚀 Momentum Breakout
        **Best for:** Strong trending stocks breaking out to new highs

        | Condition | Threshold |
        |-----------|-----------|
        | Price above EMA20 | ✅ Required |
        | EMA20 above EMA50 | ✅ Required |
        | RSI | 55–75 |
        | MACD | Bullish crossover |
        | Volume | > 1.5× 20-day average |
        | Price | Breaking above 20-day high |

        **Trade management:** Enter above resistance. SL = 1.5×ATR below entry. Target = 2R and 3R.

        ---

        ### Strategy 2 — 📉 Pullback in Uptrend
        **Best for:** Buying dips in confirmed uptrends

        | Condition | Threshold |
        |-----------|-----------|
        | Price above EMA50 | ✅ Required |
        | Near EMA20 | Within 2% |
        | RSI bounce | Was below 48, now recovering above 45 |
        | Candle | Hammer or bullish engulfing |

        **Trade management:** Enter at bounce candle close. SL = 1.2×ATR. Target = prior swing high.

        ---

        ### Strategy 3 — 🔳 Volatility Squeeze Breakout
        **Best for:** Stocks consolidating before a big move

        | Condition | Threshold |
        |-----------|-----------|
        | Bollinger Band squeeze | BB Width < 20-day avg |
        | Volume expansion | > 1.3× average |
        | Price breaking out | Above upper Bollinger Band |
        | RSI | > 50 |

        **Trade management:** Enter on breakout candle close. Tight SL inside the squeeze.

        ---

        ### Strategy 4 — 🔄 Reversal Swing
        **Best for:** Oversold stocks bouncing from extreme lows

        | Condition | Threshold |
        |-----------|-----------|
        | RSI oversold recovery | Was below 35, now above 35 |
        | MACD crossover | MACD crosses above signal |
        | Reversal candle | Hammer or bullish engulfing |

        **Trade management:** Enter at confirmation candle. SL below the swing low. Target = 2R.

        ---

        ### Backtesting Your Strategies
        Use the **📋 Backtester** page to test any strategy on any stock over a custom date range.
        Review the equity curve, win rate, profit factor, and trade log before risking real capital.
        """)

    with tab6:
        st.subheader("🔔 Alerts, Watchlist, and BSE vs NSE")
        st.markdown("""
        ### BSE vs NSE — What's the Difference?

        | Feature | NSE | BSE |
        |---------|-----|-----|
        | Symbol suffix | `.NS` | `.BO` |
        | Index | NIFTY 50, 100, 500 | SENSEX 30, BSE 100, BSE MidCap |
        | Liquidity | Generally higher | Varies by stock |
        | Options data | Available (NIFTY/BankNIFTY) | Limited |
        | Coverage | ~1800 active stocks | 5000+ stocks (many small/mid cap) |

        > For most large-cap stocks (Reliance, TCS, HDFC Bank), prices on NSE and BSE
        > are virtually identical. BSE is most useful for accessing **BSE-only small/mid-cap stocks**
        > not actively traded on NSE.

        **Same stock, both exchanges example:**
        - `RELIANCE.NS` — Reliance on NSE
        - `RELIANCE.BO` — Reliance on BSE
        - Prices will be within paise of each other

        ---

        ### Watchlist Tips
        - Add symbols **without suffix** (e.g. `INFY`) — the system defaults to `.NS` when scanning.
        - For BSE-only stocks, add with `.BO` suffix (e.g. `ZENTEC.BO`).
        - Use **⭐ Scan Watchlist** for a fast daily check on your tracked stocks.

        ---

        ### Alert Types

        | Alert Type | Triggers When |
        |------------|--------------|
        | **Price Above** | Stock price rises above your target level |
        | **Price Below** | Stock price falls below your stop level |
        | **AI Score ≥ 80** | Stock AI score crosses the strong-buy threshold |
        | **Breakout** | Stock breaks above its 20-day high |
        | **Strategy Trigger** | Any strategy fires for the stock |

        ---

        ### Telegram Setup (Step-by-Step)
        1. Open Telegram and search for **@BotFather**
        2. Send `/newbot` and follow the prompts to name your bot
        3. Copy the **bot token** provided (looks like `123456:ABC-DEF1234...`)
        4. Search for **@userinfobot** and start it — it will tell you your **chat ID**
        5. In the **🔔 Alerts** page, expand "Configure Telegram Bot"
        6. Paste your token and chat ID, then click **Test Connection**
        7. For persistent alerts, set environment variables before launching:
        ```bash
        export TELEGRAM_BOT_TOKEN="your_token"
        export TELEGRAM_CHAT_ID="your_chat_id"
        streamlit run app.py
        ```

        ---

        ### FII/DII Data
        - FII (Foreign Institutional Investors) data is fetched from the NSE API.
        - If NSE blocks the request, realistic mock data is used for demonstration.
        - FII buying + a stock breakout = **institutional participation flag** in the scanner.

        ---

        ### Options Chain
        - Works best with index symbols: `NIFTY`, `BANKNIFTY`, `MIDCPNIFTY`
        - Also works with individual large-cap stocks: `RELIANCE`, `TCS`, etc.
        - **PCR > 1.2** = bullish sentiment | **PCR < 0.7** = bearish sentiment
        - **Max Pain** = strike where option writers (usually institutions) lose least money.
          Price often gravitates toward max pain near expiry.

        ---

        ### FAQ

        **Q: The scanner shows no results. Why?**
        A: Try lowering the **Min AI Score** slider in the sidebar to 30 or 0 to see all results.

        **Q: Data is stale / not updating.**
        A: The platform caches data for 5 minutes. Click **🔄 Refresh** or wait 5 minutes and re-scan.

        **Q: Some BSE stocks show no data.**
        A: A few BSE-only symbols may not be supported by yfinance. Try the `.BO` suffix version in Stock Analysis.

        **Q: AI Model accuracy is low.**
        A: Try training on a larger universe (NIFTY100) with a longer period (2y). Also, market conditions change — retrain monthly.

        **Q: Can I run this on multiple monitors or share it?**
        A: Yes. Streamlit runs a local web server. You can access it from any device on your local network at `http://YOUR_IP:8501`.
        """)

    # Bottom disclaimer
    st.divider()
    st.markdown("""
    <div style="background:#161b22; border-radius:10px; padding:16px; border:1px solid #30363d; margin-top:20px;">
        <p style="color:#8b949e; font-size:0.85rem; margin:0;">
        ⚠️ <b>Disclaimer:</b> AI Swing Trader Pro is for <b>educational and research purposes only</b>.
        It does not constitute financial advice or a recommendation to buy or sell any security.
        Always conduct your own due diligence and consult a SEBI-registered financial advisor
        before making investment decisions. Past performance of strategies does not guarantee future results.
        The authors are not responsible for any financial losses incurred while using this platform.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Router ──────────────────────────────────────────────────────────────

if st.session_state.get("nav_override"):
    nav = st.session_state.pop("nav_override")
else:
    nav = page

if "Dashboard" in nav:
    page_dashboard()
elif "Scanner" in nav:
    page_scanner()
elif "Stock Analysis" in nav:
    page_stock_analysis()
elif "Institutional" in nav:
    page_institutional()
elif "Options" in nav:
    page_options()
elif "AI Model" in nav:
    page_ai_model()
elif "Backtester" in nav:
    page_backtester()
elif "Watchlist" in nav:
    page_watchlist()
elif "Alerts" in nav:
    page_alerts()
elif "Settings" in nav:
    page_settings()
elif "Help" in nav:
    page_help()
