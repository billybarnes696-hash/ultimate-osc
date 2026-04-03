
import io
import traceback
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from ultimate_oscillator_upgrade import (
    UltimateOscillator,
    QuantBacktestEngine,
    summarize_current_signal,
)

st.set_page_config(page_title="Ultimate Oscillator", layout="wide")


# ============================================================================
# SAFE HELPERS
# ============================================================================

@st.cache_data(show_spinner=False)
def fetch_price_history(symbol: str, period: str = "10y") -> pd.Series:
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}")

    if "Close" not in df.columns:
        raise ValueError(f"Close column missing for {symbol}")

    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    s.name = "Close"
    return s


@st.cache_data(show_spinner=False)
def parse_uploaded_price_csv(file_bytes: bytes, filename: str) -> pd.Series:
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols = {c.lower().strip(): c for c in df.columns}

    date_col = None
    close_col = None

    for candidate in ["date", "datetime", "timestamp"]:
        if candidate in cols:
            date_col = cols[candidate]
            break

    for candidate in ["close", "adj close", "adj_close", "price"]:
        if candidate in cols:
            close_col = cols[candidate]
            break

    if date_col is None or close_col is None:
        raise ValueError(
            f"{filename}: CSV must include a date column and a close/price column."
        )

    out = df[[date_col, close_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[close_col] = pd.to_numeric(out[close_col], errors="coerce")
    out = out.dropna().sort_values(date_col)
    out = out.drop_duplicates(subset=[date_col], keep="last")
    s = out.set_index(date_col)[close_col]
    s.name = "Close"
    return s


@st.cache_data(show_spinner=False)
def build_model(
    price_bytes: Optional[bytes],
    price_filename: Optional[str],
    symbol: str,
    period: str,
    zip_bytes: Optional[bytes],
    lookback: int,
    smooth_price: float,
    smooth_breadth: float,
    breadth_weight: float,
):
    if price_bytes is not None:
        price = parse_uploaded_price_csv(price_bytes, price_filename or "uploaded_price.csv")
    else:
        price = fetch_price_history(symbol, period)

    price = price.sort_index()

    oscillator = UltimateOscillator(
        lookback=lookback,
        smooth_price=smooth_price,
        smooth_breadth=smooth_breadth,
        breadth_weight=breadth_weight,
        price_weight=1.0 - breadth_weight,
    )

    osc_df = oscillator.blend_final_oscillator(
        price=price,
        zip_bytes=zip_bytes,
        snapshot_map=None,
        snap_date=None,
    ).dropna(subset=["price"])

    backtest = QuantBacktestEngine().run_backtest(osc_df)
    summary = summarize_current_signal(osc_df)

    return osc_df, backtest, summary


def make_oscillator_chart(df: pd.DataFrame, view_mode: str, lookback_bars: int) -> go.Figure:
    plot_df = df.copy()
    if lookback_bars > 0 and len(plot_df) > lookback_bars:
        plot_df = plot_df.iloc[-lookback_bars:].copy()

    fig = go.Figure()

    if view_mode in ("Combined", "All"):
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["final_osc"],
                mode="lines",
                name="Final Oscillator",
                line=dict(width=3),
            )
        )

    if view_mode in ("Breadth", "All") and "breadth_curve" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["breadth_curve"],
                mode="lines",
                name="Breadth Curve",
                line=dict(width=2, dash="dot"),
            )
        )

    if view_mode in ("Price", "All") and "price_curve" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["price_curve"],
                mode="lines",
                name="Price Curve",
                line=dict(width=2, dash="dash"),
            )
        )

    # Zones
    fig.add_hrect(y0=0, y1=20, opacity=0.12, line_width=0)
    fig.add_hrect(y0=20, y1=40, opacity=0.08, line_width=0)
    fig.add_hrect(y0=40, y1=60, opacity=0.05, line_width=0)
    fig.add_hrect(y0=60, y1=80, opacity=0.08, line_width=0)
    fig.add_hrect(y0=80, y1=100, opacity=0.12, line_width=0)

    last_x = plot_df.index[-1]
    if "final_osc" in plot_df.columns and pd.notna(plot_df["final_osc"].iloc[-1]):
        fig.add_trace(
            go.Scatter(
                x=[last_x],
                y=[plot_df["final_osc"].iloc[-1]],
                mode="markers",
                name="Current",
                marker=dict(size=12, symbol="diamond"),
            )
        )

    fig.update_layout(
        title="Ultimate Oscillator",
        height=520,
        yaxis_title="0–100 Score",
        xaxis_title="Date",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_price_chart(df: pd.DataFrame, lookback_bars: int) -> go.Figure:
    plot_df = df.copy()
    if lookback_bars > 0 and len(plot_df) > lookback_bars:
        plot_df = plot_df.iloc[-lookback_bars:].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["price"],
            mode="lines",
            name="Price",
            line=dict(width=2),
        )
    )
    fig.update_layout(
        title="Price",
        height=350,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_equity_chart(backtest, index) -> go.Figure:
    fig = go.Figure()
    if backtest.equity_curve is not None and not backtest.equity_curve.empty:
        fig.add_trace(
            go.Scatter(
                x=backtest.equity_curve.index,
                y=backtest.equity_curve.values,
                mode="lines",
                name="Strategy Equity",
                line=dict(width=2),
            )
        )
    fig.update_layout(
        title="Backtest Equity Curve",
        height=350,
        xaxis_title="Date",
        yaxis_title="Equity",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def bars_from_window(window_label: str) -> int:
    mapping = {
        "3M": 63,
        "6M": 126,
        "1Y": 252,
        "2Y": 504,
        "5Y": 1260,
        "10Y": 2520,
        "Max": 0,
    }
    return mapping.get(window_label, 252)


# ============================================================================
# UI
# ============================================================================

st.title("Ultimate Oscillator")
st.caption("Safe-boot Streamlit build for RSP / breadth oscillator work.")

with st.sidebar:
    st.header("Inputs")

    price_mode = st.radio("Price source", ["Auto-download", "Upload CSV"], index=0)

    symbol = st.text_input("Ticker", value="RSP")
    period = st.selectbox("Auto-download history", ["2y", "5y", "10y", "max"], index=2)

    price_file = None
    if price_mode == "Upload CSV":
        price_file = st.file_uploader(
            "Upload price CSV",
            type=["csv"],
            help="Needs a date column and a close/price column.",
        )

    breadth_zip = st.file_uploader(
        "Upload breadth ZIP",
        type=["zip"],
        help="StockCharts history ZIP for breadth indicators.",
    )

    st.header("Model")
    lookback = st.slider("Bell curve lookback", 63, 252, 126, step=21)
    smooth_price = st.slider("Price smoothing", 0.0, 8.0, 4.0, step=0.5)
    smooth_breadth = st.slider("Breadth smoothing", 0.0, 8.0, 4.0, step=0.5)
    breadth_weight = st.slider("Breadth weight", 0.0, 1.0, 0.70, step=0.05)

    view_mode = st.selectbox("Oscillator view", ["Combined", "Breadth", "Price", "All"], index=0)
    chart_window = st.selectbox("Chart window", ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "Max"], index=2)

    run = st.button("Build oscillator", type="primary", use_container_width=True)

st.info(
    "This version is designed not to hang on startup. Nothing heavy runs until you click "
    "**Build oscillator**."
)

if not run:
    st.stop()

try:
    price_bytes = price_file.getvalue() if price_file is not None else None
    price_filename = price_file.name if price_file is not None else None
    zip_bytes = breadth_zip.getvalue() if breadth_zip is not None else None

    with st.spinner("Building model..."):
        osc_df, backtest, summary = build_model(
            price_bytes=price_bytes,
            price_filename=price_filename,
            symbol=symbol,
            period=period,
            zip_bytes=zip_bytes,
            lookback=lookback,
            smooth_price=smooth_price,
            smooth_breadth=smooth_breadth,
            breadth_weight=breadth_weight,
        )

    if osc_df.empty:
        st.error("No model output was produced.")
        st.stop()

    bars = bars_from_window(chart_window)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("State", str(summary.get("state", "N/A")))
    c2.metric("Regime", str(summary.get("regime", "N/A")))
    c3.metric("Final Osc", f"{summary.get('final_osc', float('nan')):.2f}")
    c4.metric("Quality", f"{summary.get('quality_score', float('nan')):.2f}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Thrust", f"{summary.get('thrust_score', float('nan')):.2f}")
    d2.metric("Long Enter", f"{summary.get('long_enter_threshold', float('nan')):.2f}")
    d3.metric("Long Exit", f"{summary.get('long_exit_threshold', float('nan')):.2f}")

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Backtest", "Data"])

    with tab1:
        st.plotly_chart(make_oscillator_chart(osc_df, view_mode=view_mode, lookback_bars=bars), use_container_width=True)
        st.plotly_chart(make_price_chart(osc_df, lookback_bars=bars), use_container_width=True)

    with tab2:
        m1, m2, m3, m4, m5 = st.columns(5)
        metrics = backtest.metrics if backtest.metrics else {}
        m1.metric("Total Return", f"{metrics.get('total_return', float('nan')):.2%}" if metrics else "N/A")
        m2.metric("Annual Return", f"{metrics.get('annual_return', float('nan')):.2%}" if metrics else "N/A")
        m3.metric("Sharpe", f"{metrics.get('sharpe', float('nan')):.2f}" if metrics else "N/A")
        m4.metric("Max DD", f"{metrics.get('max_drawdown', float('nan')):.2%}" if metrics else "N/A")
        m5.metric("Trades", f"{int(metrics.get('trade_count', 0))}" if metrics else "0")

        st.plotly_chart(make_equity_chart(backtest, osc_df.index), use_container_width=True)

        if backtest.trades is not None and not backtest.trades.empty:
            st.subheader("Trades")
            st.dataframe(backtest.trades, use_container_width=True)
        else:
            st.caption("No trades generated with the current settings.")

    with tab3:
        st.subheader("Latest rows")
        st.dataframe(osc_df.tail(50), use_container_width=True)

except Exception as e:
    st.error(f"App failed: {e}")
    st.code(traceback.format_exc())
