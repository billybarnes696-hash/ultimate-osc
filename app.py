import io
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="RSP Ultimate Breadth Oscillator Lab", layout="wide")

# -----------------------------
# Utility / parsing
# -----------------------------

def normalize_symbol(name: str) -> str:
    base = name.split("/")[-1].split("\\")[-1]
    base = base.replace(".csv", "")
    return base.strip()


def parse_stockcharts_history_bytes(raw: bytes, fallback_name: str) -> pd.DataFrame:
    text = raw.decode("utf-8-sig", errors="ignore").replace("\r\n", "\n")
    lines = [line.rstrip() for line in text.split("\n") if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"File {fallback_name} does not look like a StockCharts history export.")

    symbol = lines[0].split(",")[0].strip().strip('"') or fallback_name
    header = [c.strip().strip('"') for c in lines[1].split(",")]
    rows = []
    for line in lines[2:]:
        parts = [c.strip().strip('"') for c in line.split(",")]
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        rows.append(parts[: len(header)])

    df = pd.DataFrame(rows, columns=header)
    if "Date" not in df.columns:
        raise ValueError(f"File {fallback_name} is missing a Date column.")

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).set_index("Date")
    df.attrs["symbol"] = symbol
    return df


@st.cache_data(show_spinner=False)
def load_breadth_zip(upload_bytes: bytes) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(upload_bytes)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            try:
                raw = zf.read(name)
                df = parse_stockcharts_history_bytes(raw, name)
                symbol = normalize_symbol(name)
                datasets[symbol] = df
            except Exception:
                continue
    return datasets


@st.cache_data(show_spinner=False)
def load_snapshot_csv(upload_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(upload_bytes))
    df.columns = [c.strip() for c in df.columns]
    if "Symbol" in df.columns:
        df["Symbol"] = df["Symbol"].astype(str).str.strip()
        df = df.set_index("Symbol")
    return df


@st.cache_data(show_spinner=False)
def fetch_price_history(symbol: str, years: int = 25, interval: str = "1d") -> pd.DataFrame:
    period = f"{years}y"
    if interval in {"1wk", "1mo"}:
        period = "max"
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]].dropna(how="all")


# -----------------------------
# Indicators
# -----------------------------

def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    rs = up.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / down.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return 100 - (100 / (1 + rs))


def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(length).mean()
    mad = (tp - sma).abs().rolling(length).mean()
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_len: int = 14, d_len: int = 3, smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
    ll = low.rolling(k_len).min()
    hh = high.rolling(k_len).max()
    raw_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_len).mean()
    return k, d


def bb_percent(close: pd.Series, length: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return (close - lower) / (upper - lower).replace(0, np.nan)


def tsi(close: pd.Series, long_len: int = 25, short_len: int = 13, signal_len: int = 7) -> Tuple[pd.Series, pd.Series]:
    m = close.diff()
    a = m.abs()
    tsi_line = 100 * ema(ema(m, long_len), short_len) / ema(ema(a, long_len), short_len)
    signal = ema(tsi_line, signal_len)
    return tsi_line, signal


def zscore(series: pd.Series, length: int = 126) -> pd.Series:
    mean = series.rolling(length).mean()
    std = series.rolling(length).std()
    return (series - mean) / std.replace(0, np.nan)


def to_0_100(series: pd.Series, lower: float = -3.0, upper: float = 3.0) -> pd.Series:
    x = series.clip(lower, upper)
    return 100 * (x - lower) / (upper - lower)


def smooth_osc(series: pd.Series, fast: int = 5, slow: int = 3) -> Tuple[pd.Series, pd.Series]:
    main = ema(series, fast)
    signal = ema(main, slow)
    return main.clip(0, 100), signal.clip(0, 100)


# -----------------------------
# Feature engineering
# -----------------------------
BREADTH_DEFAULTS = [
    "_Bpspx", "_Bpnya", "_nymo", "_nySI", "_nyad", "_spxa50r", "_trin", "_cpce", "_vix"
]


def score_from_series(series: pd.Series, method: str = "z", invert: bool = False, clip: float = 3.0) -> pd.Series:
    if method == "percentile":
        base = series.rolling(252, min_periods=60).rank(pct=True) * 100
        score = base
    else:
        zs = zscore(series, 126)
        score = to_0_100(zs, lower=-clip, upper=clip)
    if invert:
        score = 100 - score
    return score


def build_breadth_scores(datasets: Dict[str, pd.DataFrame], selected: List[str]) -> pd.DataFrame:
    pieces = []
    for key in selected:
        if key not in datasets:
            continue
        close = datasets[key]["Close"].copy()
        invert = key.lower() in {"_trin", "_cpce", "_vix", "vxx", "spxs_svol"}
        score = score_from_series(close, method="z", invert=invert)
        pieces.append(score.rename(key))
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=1).sort_index()
    out["breadth_raw"] = out.mean(axis=1, skipna=True)
    out["breadth_main"], out["breadth_signal"] = smooth_osc(out["breadth_raw"], fast=5, slow=3)
    return out


def build_price_scores(price: pd.DataFrame, timeframe_name: str = "daily") -> pd.DataFrame:
    close = price["Close"]
    high = price["High"] if "High" in price else close
    low = price["Low"] if "Low" in price else close

    out = pd.DataFrame(index=price.index)
    out["rsi3"] = rsi(close, 3)
    out["rsi14"] = rsi(close, 14)
    out["cci14"] = cci(high, low, close, 14)
    out["bbp20"] = bb_percent(close, 20)
    k, d = stochastic(high, low, close, 14, 3, 3)
    out["stoch_k"] = k
    out["stoch_d"] = d
    tsi_line, tsi_sig = tsi(close, 25, 13, 7)
    out["tsi"] = tsi_line
    out["tsi_sig"] = tsi_sig

    score_cols = []
    out["score_rsi3"] = (100 - (out["rsi3"] - 50).abs() * 2).clip(0, 100)
    score_cols.append("score_rsi3")
    out["score_cci14"] = to_0_100(-zscore(out["cci14"], 126), lower=-3, upper=3)
    score_cols.append("score_cci14")
    out["score_bbp20"] = ((1 - out["bbp20"].clip(-0.5, 1.5)) * 50).clip(0, 100)
    score_cols.append("score_bbp20")
    out["score_stoch"] = (100 - out["stoch_k"]).clip(0, 100)
    score_cols.append("score_stoch")
    out["score_tsi"] = to_0_100(-zscore(out["tsi"], 126), lower=-3, upper=3)
    score_cols.append("score_tsi")

    out["price_raw"] = out[score_cols].mean(axis=1, skipna=True)
    out["price_main"], out["price_signal"] = smooth_osc(out["price_raw"], fast=5, slow=3)
    return out


def build_ultimate_oscillator(price_df: pd.DataFrame, breadth_scores: Optional[pd.DataFrame], breadth_weight: float = 0.45) -> pd.DataFrame:
    price_scores = build_price_scores(price_df)
    combined = price_scores[["price_main"]].rename(columns={"price_main": "price_score"}).copy()
    if breadth_scores is not None and not breadth_scores.empty:
        combined = combined.join(breadth_scores[["breadth_main"]].rename(columns={"breadth_main": "breadth_score"}), how="left")
    else:
        combined["breadth_score"] = np.nan

    bw = float(np.clip(breadth_weight, 0.0, 0.9))
    pw = 1.0 - bw
    combined["ultimate_raw"] = pw * combined["price_score"] + bw * combined["breadth_score"].fillna(combined["price_score"])
    combined["ultimate_main"], combined["ultimate_signal"] = smooth_osc(combined["ultimate_raw"], fast=5, slow=3)
    return combined.join(price_scores, how="left")


# -----------------------------
# Backtest / Monte Carlo
# -----------------------------

def compute_trade_metrics(trade_returns: pd.Series) -> Dict[str, float]:
    trade_returns = trade_returns.dropna()
    if trade_returns.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "median_return": np.nan,
            "cum_return": np.nan,
            "max_drawdown": np.nan,
            "profit_factor": np.nan,
        }
    equity = (1 + trade_returns).cumprod()
    dd = equity / equity.cummax() - 1
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = -trade_returns[trade_returns < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.nan
    return {
        "trades": float(len(trade_returns)),
        "win_rate": float((trade_returns > 0).mean()),
        "avg_return": float(trade_returns.mean()),
        "median_return": float(trade_returns.median()),
        "cum_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(dd.min()),
        "profit_factor": float(pf) if pd.notna(pf) else np.nan,
    }


def backtest_threshold_strategy(df: pd.DataFrame, long_level: int, short_level: int, hold_bars: int, mode: str = "long_short") -> Tuple[pd.Series, pd.DataFrame]:
    data = df.copy()
    osc = data["ultimate_main"]
    sig = data["ultimate_signal"]
    fut_ret = data["Close"].shift(-hold_bars) / data["Close"] - 1

    long_entry = (osc.shift(1) <= long_level) & (osc > long_level) & (osc > sig)
    short_entry = (osc.shift(1) >= short_level) & (osc < short_level) & (osc < sig)

    trade_rows = []
    long_returns = fut_ret.where(long_entry)
    if mode in {"long_short", "long_only", "long_cash"}:
        for idx in data.index[long_entry.fillna(False)]:
            exit_idx_pos = data.index.get_loc(idx) + hold_bars
            if exit_idx_pos >= len(data.index):
                continue
            exit_dt = data.index[exit_idx_pos]
            ret = fut_ret.loc[idx]
            trade_rows.append({"entry_date": idx, "exit_date": exit_dt, "side": "LONG", "return": ret})

    short_returns = -fut_ret.where(short_entry)
    if mode == "long_short":
        for idx in data.index[short_entry.fillna(False)]:
            exit_idx_pos = data.index.get_loc(idx) + hold_bars
            if exit_idx_pos >= len(data.index):
                continue
            exit_dt = data.index[exit_idx_pos]
            ret = -fut_ret.loc[idx]
            trade_rows.append({"entry_date": idx, "exit_date": exit_dt, "side": "SHORT", "return": ret})

    if mode == "long_only":
        trade_returns = long_returns.dropna()
    elif mode == "long_cash":
        trade_returns = long_returns.dropna()
    else:
        trade_returns = pd.concat([long_returns, short_returns]).sort_index().dropna()

    trades_df = pd.DataFrame(trade_rows)
    return trade_returns, trades_df


def run_grid_search(df: pd.DataFrame, long_levels: List[int], short_levels: List[int], hold_bars_list: List[int], mode: str) -> pd.DataFrame:
    rows = []
    for long_level in long_levels:
        for short_level in short_levels:
            if long_level >= short_level:
                continue
            for hold_bars in hold_bars_list:
                rets, trades = backtest_threshold_strategy(df, long_level, short_level, hold_bars, mode=mode)
                metrics = compute_trade_metrics(rets)
                metrics.update({
                    "long_level": long_level,
                    "short_level": short_level,
                    "hold_bars": hold_bars,
                    "mode": mode,
                })
                if metrics["trades"] > 0:
                    # blended robustness-ish score
                    wr = metrics["win_rate"] if pd.notna(metrics["win_rate"]) else 0
                    avg_r = metrics["avg_return"] if pd.notna(metrics["avg_return"]) else 0
                    mdd = abs(metrics["max_drawdown"]) if pd.notna(metrics["max_drawdown"]) else 1
                    pf = metrics["profit_factor"] if pd.notna(metrics["profit_factor"]) else 1
                    metrics["score"] = (avg_r * 1000) + (wr * 25) + (pf * 5) - (mdd * 20) + min(metrics["trades"], 300) / 50
                else:
                    metrics["score"] = np.nan
                rows.append(metrics)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["score", "avg_return", "win_rate"], ascending=False)
    return out


def monte_carlo_from_trades(trade_returns: pd.Series, n_sims: int = 1000, seed: int = 42) -> pd.DataFrame:
    trade_returns = trade_returns.dropna().astype(float)
    if trade_returns.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    vals = trade_returns.values
    sims = []
    n = len(vals)
    for _ in range(n_sims):
        sampled = rng.choice(vals, size=n, replace=True)
        equity = np.cumprod(1 + sampled)
        dd = equity / np.maximum.accumulate(equity) - 1
        sims.append({
            "cum_return": equity[-1] - 1,
            "max_drawdown": dd.min(),
            "avg_return": sampled.mean(),
            "win_rate": (sampled > 0).mean(),
        })
    return pd.DataFrame(sims)


# -----------------------------
# Visualization
# -----------------------------

def plot_oscillator(df: pd.DataFrame, title: str, price_name: str = "Close") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[price_name], name="RSP Price", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ultimate_main"], name="Ultimate", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ultimate_signal"], name="Signal", yaxis="y2"))
    fig.add_hline(y=20, line_dash="dot", opacity=0.4, yref="y2")
    fig.add_hline(y=80, line_dash="dot", opacity=0.4, yref="y2")
    fig.update_layout(
        title=title,
        xaxis=dict(domain=[0.0, 1.0]),
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(title="Oscillator", overlaying="y", side="right", range=[0, 100]),
        height=520,
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# -----------------------------
# App
# -----------------------------
st.title("RSP Ultimate Breadth Oscillator Lab")
st.caption("Backtest sweet spots for RSP + breadth, then build a smooth slow-stoch-style composite and stress test it.")

with st.sidebar:
    st.header("Inputs")
    symbol = st.text_input("ETF symbol", value="RSP")
    lookback_years = st.slider("Historical lookback (years)", 5, 25, 25)
    breadth_weight = st.slider("Breadth weight in ultimate oscillator", 0.0, 0.9, 0.45, 0.05)
    mode = st.selectbox("Backtest mode", ["long_short", "long_only", "long_cash"], index=0)
    long_levels = st.multiselect("Long trigger candidates", options=list(range(10, 41, 2)), default=[20, 22, 24, 26, 28, 30])
    short_levels = st.multiselect("Short trigger candidates", options=list(range(60, 91, 2)), default=[70, 72, 74, 76, 78, 80])
    hold_bars = st.multiselect("Hold bars", options=[1, 2, 3, 5, 8, 10, 15, 20], default=[3, 5, 8])

    st.markdown("---")
    breadth_zip_file = st.file_uploader("Upload breadth bucket zip", type=["zip"])
    snapshot_file = st.file_uploader("Upload SC snapshot csv", type=["csv"])

    st.markdown("**Breadth symbols**")
    breadth_symbol_defaults = BREADTH_DEFAULTS.copy()

# Allow local uploaded files from current environment as defaults for convenience.
if breadth_zip_file is None:
    try:
        with open("/mnt/data/breadth.zip", "rb") as f:
            breadth_zip_bytes = f.read()
    except Exception:
        breadth_zip_bytes = None
else:
    breadth_zip_bytes = breadth_zip_file.getvalue()

if snapshot_file is None:
    try:
        with open("/mnt/data/SC (20).csv", "rb") as f:
            snapshot_bytes = f.read()
    except Exception:
        snapshot_bytes = None
else:
    snapshot_bytes = snapshot_file.getvalue()

breadth_datasets = load_breadth_zip(breadth_zip_bytes) if breadth_zip_bytes else {}
snapshot_df = load_snapshot_csv(snapshot_bytes) if snapshot_bytes else pd.DataFrame()

available_breadth = sorted(breadth_datasets.keys())
selected_breadth = st.sidebar.multiselect(
    "Historical breadth series to use",
    options=available_breadth,
    default=[x for x in breadth_symbol_defaults if x in available_breadth],
)

price_daily = fetch_price_history(symbol, years=lookback_years, interval="1d")
price_hourly = fetch_price_history(symbol, years=min(lookback_years, 2), interval="60m")

if price_daily.empty:
    st.error(f"Could not pull history for {symbol}. Check symbol or network access in Streamlit runtime.")
    st.stop()

# Build breadth scores and align with daily price.
breadth_scores_daily = build_breadth_scores(breadth_datasets, selected_breadth)
ultimate_daily = build_ultimate_oscillator(price_daily, breadth_scores_daily, breadth_weight=breadth_weight)
ultimate_daily = ultimate_daily.join(price_daily[["Close"]], how="left")

weekly_price = price_daily.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
breadth_weekly = breadth_scores_daily[["breadth_main", "breadth_signal"]].resample("W-FRI").last() if not breadth_scores_daily.empty else pd.DataFrame()
ultimate_weekly = build_ultimate_oscillator(weekly_price, breadth_weekly, breadth_weight=breadth_weight)
ultimate_weekly = ultimate_weekly.join(weekly_price[["Close"]], how="left")

hourly_note = None
if price_hourly.empty:
    ultimate_hourly = pd.DataFrame()
    hourly_note = "Hourly history was unavailable from the data source."
else:
    # Historical hourly breadth is not present in the uploaded bucket, so this is price-led hourly only.
    ultimate_hourly = build_ultimate_oscillator(price_hourly, None, breadth_weight=0.0)
    ultimate_hourly = ultimate_hourly.join(price_hourly[["Close"]], how="left")
    hourly_note = "Hourly oscillator is currently price-led only. The uploaded breadth bucket is daily history, so there is no historical hourly breadth backtest yet."

# Snapshot summary
current_snapshot = {}
if not snapshot_df.empty:
    for sym in ["RSP", "$NYMO", "$NYSI", "$BPSPX", "$SPXA50R", "$NYAD", "$TRIN", "$CPCE", "$VIX"]:
        if sym in snapshot_df.index:
            current_snapshot[sym] = dict(snapshot_df.loc[sym])

summary_cols = st.columns(4)
summary_cols[0].metric("RSP daily history start", str(price_daily.index.min().date()))
summary_cols[1].metric("RSP daily history end", str(price_daily.index.max().date()))
summary_cols[2].metric("Breadth files loaded", len(breadth_datasets))
summary_cols[3].metric("Selected breadth series", len(selected_breadth))

if current_snapshot:
    st.subheader("Latest snapshot readings (from SC file)")
    snap_show = pd.DataFrame(current_snapshot).T[[c for c in ["Close", "Daily PctChange(1,Daily Close)"] if c in pd.DataFrame(current_snapshot).T.columns]]
    st.dataframe(snap_show, use_container_width=True)

# Backtest tab
research_tab, mc_tab, views_tab, data_tab = st.tabs(["Backtest Lab", "Monte Carlo", "Hourly / Daily / Weekly", "Data Review"])

with research_tab:
    st.subheader("Backtest combinations on the smooth ultimate oscillator")
    st.write("This grid tests which slow-stoch-style trigger levels work best on the daily ultimate oscillator for RSP. Use the results to select the sweet-spot long/short bands for the final visual.")
    results = run_grid_search(ultimate_daily.join(price_daily[["Close"]]), sorted(long_levels), sorted(short_levels), sorted(hold_bars), mode=mode)
    if results.empty:
        st.warning("No valid backtest results were produced with the selected settings.")
    else:
        st.dataframe(results.head(50), use_container_width=True)
        top = results.iloc[0]
        st.success(
            f"Current top combo: long cross above {int(top['long_level'])}, short cross below {int(top['short_level'])}, hold {int(top['hold_bars'])} bars, mode={top['mode']}."
        )
        best_rets, best_trades = backtest_threshold_strategy(
            ultimate_daily.join(price_daily[["Close"]]), int(top["long_level"]), int(top["short_level"]), int(top["hold_bars"]), mode=mode
        )
        c1, c2, c3, c4 = st.columns(4)
        met = compute_trade_metrics(best_rets)
        c1.metric("Trades", int(met["trades"]))
        c2.metric("Win rate", f"{met['win_rate']:.1%}" if pd.notna(met["win_rate"]) else "n/a")
        c3.metric("Avg trade", f"{met['avg_return']:.2%}" if pd.notna(met["avg_return"]) else "n/a")
        c4.metric("Max DD", f"{met['max_drawdown']:.2%}" if pd.notna(met["max_drawdown"]) else "n/a")
        st.plotly_chart(plot_oscillator(ultimate_daily[["Close", "ultimate_main", "ultimate_signal"]].dropna().tail(1000), "Daily Ultimate Oscillator (RSP)"), use_container_width=True)
        st.write("Top trade samples")
        st.dataframe(best_trades.head(20), use_container_width=True)

with mc_tab:
    st.subheader("Monte Carlo pressure testing")
    if results.empty:
        st.info("Run the backtest first.")
    else:
        top = results.iloc[0]
        best_rets, _ = backtest_threshold_strategy(
            ultimate_daily.join(price_daily[["Close"]]), int(top["long_level"]), int(top["short_level"]), int(top["hold_bars"]), mode=mode
        )
        sims = monte_carlo_from_trades(best_rets, n_sims=1000)
        if sims.empty:
            st.warning("No trades available for Monte Carlo.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Median MC return", f"{sims['cum_return'].median():.2%}")
            c2.metric("10th pct MC return", f"{sims['cum_return'].quantile(0.10):.2%}")
            c3.metric("Median MC max DD", f"{sims['max_drawdown'].median():.2%}")
            st.write("Monte Carlo summary")
            st.dataframe(sims.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T, use_container_width=True)
            fig = go.Figure()
            fig.add_histogram(x=sims["cum_return"], nbinsx=50, name="MC cumulative return")
            fig.update_layout(height=420, title="Monte Carlo Distribution of Cumulative Return")
            st.plotly_chart(fig, use_container_width=True)

with views_tab:
    st.subheader("Ultimate oscillator family")
    st.write("Daily and weekly include uploaded daily breadth history. Hourly is currently price-led because the uploaded bucket does not contain historical hourly breadth series.")
    st.plotly_chart(plot_oscillator(ultimate_daily[["Close", "ultimate_main", "ultimate_signal"]].dropna().tail(1000), "Daily Ultimate Breadth Oscillator"), use_container_width=True)
    st.plotly_chart(plot_oscillator(ultimate_weekly[["Close", "ultimate_main", "ultimate_signal"]].dropna().tail(500), "Weekly Ultimate Breadth Oscillator"), use_container_width=True)
    if not ultimate_hourly.empty:
        st.plotly_chart(plot_oscillator(ultimate_hourly[["Close", "ultimate_main", "ultimate_signal"]].dropna().tail(1000), "Hourly Tactical Oscillator (price-led for now)"), use_container_width=True)
    if hourly_note:
        st.info(hourly_note)

with data_tab:
    st.subheader("Uploaded data review")
    st.write("Breadth zip format recognized as StockCharts history export. Snapshot CSV format recognized as scan snapshot / SC20 export.")
    st.write("Detected breadth files")
    st.dataframe(pd.DataFrame({"symbol": available_breadth}), use_container_width=True)
    if not breadth_scores_daily.empty:
        st.write("Breadth score preview")
        st.dataframe(breadth_scores_daily.tail(20), use_container_width=True)
    st.write("Daily oscillator preview")
    st.dataframe(ultimate_daily.tail(20), use_container_width=True)

st.markdown("---")
st.markdown(
    "**Current implementation note:** the uploaded breadth bucket gives strong daily historical coverage for RSP, which is enough to build and backtest the daily and weekly breadth oscillator now. A fully historical hourly breadth oscillator will need historical intraday breadth inputs or an agreed proxy layer."
)
