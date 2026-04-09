import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import gaussian_kde, norm

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Diamond Scanner", layout="wide")
st.title("💎 Diamond Scanner")
st.caption(
    "Warehouse-first stock scanner with persistent caching, adjustable diamond scoring, and empirical bell curves."
)

CACHE_ROOT = Path("cache")
DAILY_CACHE = CACHE_ROOT / "daily"
META_CACHE = CACHE_ROOT / "meta"
SNAPSHOT_CACHE = CACHE_ROOT / "snapshots"
for p in [CACHE_ROOT, DAILY_CACHE, META_CACHE, SNAPSHOT_CACHE]:
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_UNIVERSE = [
    # Broad ETFs and liquid stocks
    "SPY","QQQ","IWM","DIA","RSP","SMH","XLF","XLK","XLE","XLI","XLP","XLV","XLY","XLC","XLB","XLRE","XBI","ARKK","SOXX","KRE",
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD","NFLX","AVGO","COST","JPM","GS","MS","BAC","WFC","C","SCHW","BLK",
    "UNH","LLY","ABBV","MRK","PFE","ISRG","VRTX","ABT","TMO","DHR","HD","LOW","MCD","SBUX","BKNG","UBER","ORLY","TJX",
    "CAT","DE","GE","ETN","PH","HON","MMM","UNP","CSX","NSC","LMT","RTX","BA","NOC","INTC","MU","QCOM","AMAT","LRCX","KLAC",
    "CRM","ORCL","ADBE","NOW","PANW","CRWD","PLTR","SNOW","SHOP","MDB","ANET","V","MA","PYPL","COIN","HOOD","AXP","CB","PGR",
    "LEVI","EXTR","FTV","ITT","FIS","FI","ICE","CME","KKR","BX","APO","Ares","HWM","TT","URI"
]
DEFAULT_UNIVERSE = sorted({s.upper() for s in DEFAULT_UNIVERSE if s.isalpha() or s.replace('^','').isalnum()})

# =============================================================================
# HELPERS
# =============================================================================
def sanitize_ticker(t: str) -> str:
    return str(t).strip().upper().replace("/", "-")


def parquet_path(symbol: str) -> Path:
    return DAILY_CACHE / f"{sanitize_ticker(symbol)}.parquet"


def meta_path(symbol: str) -> Path:
    return META_CACHE / f"{sanitize_ticker(symbol)}.json"


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def tsi(close: pd.Series, long: int, short: int, signal: int) -> Tuple[pd.Series, pd.Series]:
    m = close.diff()
    abs_m = m.abs()
    double_smoothed_m = ema(ema(m, long), short)
    double_smoothed_abs = ema(ema(abs_m, long), short)
    tsi_line = 100 * double_smoothed_m / double_smoothed_abs.replace(0, np.nan)
    signal_line = ema(tsi_line, signal)
    return tsi_line, signal_line


def cci(df: pd.DataFrame, length: int = 15) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def bb_pct(close: pd.Series, length: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return (close - lower) / (upper - lower).replace(0, np.nan)


def percentile_rank(series: pd.Series, lookback: int = 252) -> pd.Series:
    def _pct(x):
        s = pd.Series(x).dropna()
        if len(s) < max(20, lookback // 4):
            return np.nan
        return 100.0 * (s.rank(pct=True).iloc[-1])
    return series.rolling(lookback, min_periods=max(20, lookback // 4)).apply(_pct, raw=False)


def robust_cdf(series: pd.Series, lookback: int = 252) -> pd.Series:
    med = series.rolling(lookback, min_periods=max(20, lookback // 4)).median()
    mad = series.rolling(lookback, min_periods=max(20, lookback // 4)).apply(
        lambda x: np.median(np.abs(pd.Series(x).dropna() - np.median(pd.Series(x).dropna()))) if len(pd.Series(x).dropna()) else np.nan,
        raw=False,
    )
    robust_std = 1.4826 * mad.replace(0, np.nan)
    z = ((series - med) / robust_std).clip(-3.5, 3.5)
    return pd.Series(norm.cdf(z) * 100.0, index=series.index)


def empirical_score(series: pd.Series, lookback: int = 252, invert: bool = False, smooth: int = 3) -> pd.Series:
    p = percentile_rank(series, lookback)
    c = robust_cdf(series, lookback)
    slope = series.diff().ewm(span=5, adjust=False).mean()
    persist = 50 + np.tanh(slope.fillna(0) * 5) * 12
    score = (0.50 * p) + (0.35 * c) + (0.15 * persist)
    if invert:
        score = 100 - score
    if smooth and smooth > 1:
        score = score.ewm(span=smooth, adjust=False).mean()
    return score.clip(0, 100)


def batch_download(symbols: List[str], period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    data = yf.download(
        tickers=symbols,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return data


def normalize_download(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    symbol = sanitize_ticker(symbol)
    if data.empty:
        return pd.DataFrame()
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if symbol not in data.columns.get_level_values(0):
                return pd.DataFrame()
            out = data[symbol].copy()
        else:
            out = data.copy()
        needed = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in out.columns]
        out = out[needed].copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if "Adj Close" not in out.columns and "Close" in out.columns:
            out["Adj Close"] = out["Close"]
        return out.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def load_cached_symbol(symbol: str) -> pd.DataFrame:
    path = parquet_path(symbol)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df.sort_index()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_cached_symbol(symbol: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    path = parquet_path(symbol)
    df = df.copy().sort_index()
    df.to_parquet(path)


def symbol_is_stale(df: pd.DataFrame, refresh_days: int = 1) -> bool:
    if df.empty:
        return True
    last = pd.to_datetime(df.index.max())
    now = pd.Timestamp.now().tz_localize(None)
    if now.normalize() <= last.normalize():
        return False
    return (now.normalize() - last.normalize()).days >= refresh_days


def refresh_cache(symbols: List[str], batch_size: int, sleep_s: float, period: str = "2y") -> Tuple[int, int]:
    updated = 0
    failed = 0
    symbols = [sanitize_ticker(s) for s in symbols]
    stale = [s for s in symbols if symbol_is_stale(load_cached_symbol(s), refresh_days=1)]
    for i in range(0, len(stale), batch_size):
        batch = stale[i:i + batch_size]
        try:
            raw = batch_download(batch, period=period, interval="1d")
            for sym in batch:
                df = normalize_download(raw, sym)
                if df.empty:
                    failed += 1
                    continue
                old = load_cached_symbol(sym)
                if not old.empty:
                    df = pd.concat([old, df]).sort_index()
                    df = df[~df.index.duplicated(keep="last")]
                save_cached_symbol(sym, df)
                updated += 1
        except Exception:
            failed += len(batch)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return updated, failed


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 60:
        return pd.DataFrame()
    out = df.copy()
    close = out["Adj Close"].fillna(out["Close"])
    out["SMA10"] = close.rolling(10).mean()
    out["SMA20"] = close.rolling(20).mean()
    out["SMA50"] = close.rolling(50).mean()
    out["AVG_VOL20"] = out["Volume"].rolling(20).mean()
    out["EXT10"] = (close / out["SMA10"]) - 1
    out["TSI_323"], out["TSI_323_SIG"] = tsi(close, 3, 2, 3)
    out["TSI_747"], out["TSI_747_SIG"] = tsi(close, 7, 4, 7)
    out["CCI15"] = cci(out, 15)
    out["CCI15_DELTA"] = out["CCI15"].diff()
    out["BBPCT"] = bb_pct(close, 20, 2.0)
    out["RET5"] = close.pct_change(5)
    out["RET3"] = close.pct_change(3)
    out["TSI_323_SCORE"] = empirical_score(out["TSI_323"], 252)
    out["TSI_747_SCORE"] = empirical_score(out["TSI_747"], 252)
    out["CCI15_SCORE"] = empirical_score(out["CCI15"], 252)
    out["BBPCT_SCORE"] = empirical_score(out["BBPCT"], 252)
    out["EXT10_SCORE"] = empirical_score(out["EXT10"], 252)

    # Diamond sub-scores: strength + early rollover + stretch
    out["HEAT_SCORE"] = (
        0.33 * out["TSI_323_SCORE"] +
        0.23 * out["TSI_747_SCORE"] +
        0.22 * out["CCI15_SCORE"] +
        0.22 * out["BBPCT_SCORE"]
    )

    cci_roll_component = (50 + np.tanh((-out["CCI15_DELTA"].fillna(0)) / 8.0) * 50).clip(0, 100)
    ext_component = out["EXT10_SCORE"]
    signal_gap = (out["TSI_323"] - out["TSI_323_SIG"]).fillna(0)
    gap_component = (50 + np.tanh(signal_gap / 6.0) * 20).clip(0, 100)
    overheat_penalty = np.where(out["TSI_323"] > 99.5, 8, 0) + np.where(out["BBPCT"] > 1.10, 8, 0)

    out["DIAMOND_SCORE"] = (
        0.45 * out["HEAT_SCORE"] +
        0.25 * cci_roll_component +
        0.20 * ext_component +
        0.10 * gap_component -
        overheat_penalty
    ).clip(0, 100)

    out["ULTIMATE_SCORE"] = (
        0.30 * out["TSI_323_SCORE"] +
        0.25 * out["TSI_747_SCORE"] +
        0.20 * out["CCI15_SCORE"] +
        0.15 * out["BBPCT_SCORE"] +
        0.10 * out["EXT10_SCORE"]
    ).clip(0, 100)
    return out


def latest_row_with_symbol(symbol: str, df: pd.DataFrame) -> pd.Series:
    row = df.dropna(how="all").iloc[-1].copy()
    row["Ticker"] = symbol
    row["Date"] = df.dropna(how="all").index[-1]
    return row


def run_scan(universe: List[str], min_price: float, min_avg_vol: float, tsi323_min: float, tsi747_min: float,
             cci_min: float, bbpct_min: float, ext10_min: float, require_above_sma20: bool,
             require_above_sma50: bool, require_cci_roll: bool) -> pd.DataFrame:
    rows = []
    for symbol in universe:
        raw = load_cached_symbol(symbol)
        feat = compute_features(raw)
        if feat.empty:
            continue
        last = latest_row_with_symbol(symbol, feat)
        close = safe_float(last.get("Adj Close", last.get("Close", np.nan)))
        avg_vol = safe_float(last.get("AVG_VOL20", np.nan))
        conds = [
            close > min_price,
            avg_vol > min_avg_vol,
            safe_float(last.get("TSI_323", np.nan)) > tsi323_min,
            safe_float(last.get("TSI_747", np.nan)) > tsi747_min,
            safe_float(last.get("CCI15", np.nan)) > cci_min,
            safe_float(last.get("BBPCT", np.nan)) > bbpct_min,
            safe_float(last.get("EXT10", np.nan)) > ext10_min,
        ]
        if require_above_sma20:
            conds.append(close > safe_float(last.get("SMA20", np.nan)))
        if require_above_sma50:
            conds.append(close > safe_float(last.get("SMA50", np.nan)))
        if require_cci_roll:
            conds.append(safe_float(last.get("CCI15_DELTA", np.nan)) < 0)
        if all(conds):
            rows.append(last)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    cols = [
        "Ticker","Date","Adj Close","AVG_VOL20","SMA20","SMA50","TSI_323","TSI_747","CCI15","CCI15_DELTA",
        "BBPCT","EXT10","HEAT_SCORE","DIAMOND_SCORE","ULTIMATE_SCORE","TSI_323_SCORE","TSI_747_SCORE","CCI15_SCORE","BBPCT_SCORE","EXT10_SCORE"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].sort_values(["DIAMOND_SCORE","BBPCT"], ascending=[False, False]).reset_index(drop=True)


def empirical_bell_figure(series: pd.Series, current_value: float, prev_value: float, title: str, x_title: str = "Score (0-100)") -> go.Figure:
    fig = go.Figure()
    hist = pd.Series(series).dropna().tail(252)
    x = np.linspace(0, 100, 300)
    if len(hist) >= 20:
        try:
            kde = gaussian_kde(hist.values)
            y = kde(x)
            y = y / y.max() * 100
        except Exception:
            z = (x - 50) / 16.5
            y = norm.pdf(z)
            y = y / y.max() * 100
    else:
        z = (x - 50) / 16.5
        y = norm.pdf(z)
        y = y / y.max() * 100

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy", line=dict(width=2)))
    bands = [
        (0,15,"#00c853"),(15,30,"#90ee90"),(30,45,"#fff59d"),(45,55,"#ffffff"),(55,70,"#ffe082"),(70,85,"#ef9a9a"),(85,100,"#ff1744")
    ]
    for lo, hi, color in bands:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=0.12, line_width=0)
    fig.add_vline(x=50, line_dash="dash", line_color="gray")

    if pd.notna(prev_value):
        py = norm.pdf((prev_value - 50) / 16.5) / norm.pdf(0) * 100
        fig.add_trace(go.Scatter(x=[prev_value], y=[py], mode="markers", marker=dict(size=9, color="rgba(255,255,255,0.45)"), showlegend=False))
    if pd.notna(current_value):
        cy = norm.pdf((current_value - 50) / 16.5) / norm.pdf(0) * 100
        fig.add_trace(go.Scatter(x=[current_value], y=[cy], mode="markers+text", text=[f"{current_value:.1f}"], textposition="top center", marker=dict(size=14, line=dict(color="black", width=2)), showlegend=False))

    fig.update_layout(title=title, height=230, margin=dict(l=20,r=20,t=45,b=20), xaxis_title=x_title, yaxis_visible=False)
    fig.update_xaxes(range=[0,100])
    return fig


def price_panel(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"].fillna(df["Close"]), mode="lines", name="Price"))
    if "SMA10" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA10"], mode="lines", name="SMA10", line=dict(dash="dash")))
    if "SMA20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], mode="lines", name="SMA20", line=dict(dash="dot")))
    fig.update_layout(title=f"{symbol} Price", height=320, margin=dict(l=20,r=20,t=45,b=20))
    return fig


# =============================================================================
# APP
# =============================================================================
BUILTIN_UNIVERSES = {
    "Starter 40": DEFAULT_UNIVERSE[:40],
    "Default live universe": DEFAULT_UNIVERSE,
    "ETFs only": [s for s in DEFAULT_UNIVERSE if s in {"SPY","QQQ","IWM","DIA","RSP","SMH","XLF","XLK","XLE","XLI","XLP","XLV","XLY","XLC","XLB","XLRE","XBI","ARKK","SOXX","KRE"}],
    "Semis + AI": [s for s in DEFAULT_UNIVERSE if s in {"SMH","SOXX","NVDA","AMD","AVGO","MU","QCOM","AMAT","LRCX","KLAC","INTC","TSLA","ANET","PLTR"}],
    "Financials focus": [s for s in DEFAULT_UNIVERSE if s in {"XLF","KRE","JPM","GS","MS","BAC","WFC","C","SCHW","BLK","KKR","BX","APO","ICE","CME","AXP","CB","PGR","FIS","FI","HOOD","COIN"}],
}


SCAN_PROFILES = {
    "Custom": None,
    "Watchlist": {
        "min_price": 5.0,
        "min_avg_vol": 1_000_000.0,
        "require_above_sma20": True,
        "require_above_sma50": True,
        "tsi323_min": 93.0,
        "tsi747_min": 68.0,
        "cci_min": 85.0,
        "require_cci_roll": True,
        "bbpct_min": 0.92,
        "ext10_min": 0.028,
    },
    "Standard Diamond": {
        "min_price": 5.0,
        "min_avg_vol": 1_000_000.0,
        "require_above_sma20": True,
        "require_above_sma50": True,
        "tsi323_min": 95.0,
        "tsi747_min": 70.0,
        "cci_min": 90.0,
        "require_cci_roll": True,
        "bbpct_min": 0.95,
        "ext10_min": 0.03,
    },
    "Aggressive Fade": {
        "min_price": 5.0,
        "min_avg_vol": 1_000_000.0,
        "require_above_sma20": True,
        "require_above_sma50": True,
        "tsi323_min": 96.0,
        "tsi747_min": 72.0,
        "cci_min": 95.0,
        "require_cci_roll": True,
        "bbpct_min": 0.98,
        "ext10_min": 0.032,
    },
}

FIRE_OVERRIDES = {
    "tsi323_min": 97.0,
    "tsi747_min": 75.0,
    "cci_min": 100.0,
    "bbpct_min": 1.00,
    "ext10_min": 0.035,
    "require_cci_roll": True,
    "require_above_sma20": True,
    "require_above_sma50": True,
}


def apply_scan_profile(profile_name: str, fire_mode: bool = False) -> Dict[str, float]:
    base = SCAN_PROFILES.get(profile_name)
    if not base:
        base = SCAN_PROFILES["Standard Diamond"].copy()
    else:
        base = base.copy()
    if fire_mode:
        for key, value in FIRE_OVERRIDES.items():
            if isinstance(value, bool):
                base[key] = value
            else:
                base[key] = max(float(base.get(key, value)), float(value))
    return base

with st.sidebar:
    st.header("Universe")
    universe_mode = st.radio("Universe source", ["Built-in live universe", "Upload CSV/TXT", "Paste tickers"], index=0)
    preset_name = st.selectbox("Built-in universe preset", list(BUILTIN_UNIVERSES.keys()), index=1)
    uploaded = st.file_uploader("Optional ticker CSV/TXT (one ticker per line or a column named ticker)", type=["csv", "txt"])
    universe_text = st.text_area("Or paste tickers", value="")

    st.header("Cache refresh")
    batch_size = st.slider("Yahoo batch size", 5, 50, 25, 5)
    sleep_s = st.slider("Sleep between batches (sec)", 0.0, 2.0, 0.25, 0.05)
    refresh_period = st.selectbox("Download window", ["1y", "2y", "3y"], index=1)

    st.header("Scan logic")
    scan_profile_name = st.selectbox("Preset scan logic", list(SCAN_PROFILES.keys()), index=1)
    fire_mode = st.checkbox("Fire mode (sure-fire only)", value=False, help="Tightens the scan to only the hottest names that are already starting to cool.")
    profile_values = apply_scan_profile(scan_profile_name, fire_mode)

    st.caption("Toggle a preset first, then fine-tune below if you want.")
    min_price = st.number_input("Min price", value=float(profile_values["min_price"]), step=0.5)
    min_avg_vol = st.number_input("Min avg vol20", value=float(profile_values["min_avg_vol"]), step=100_000.0)
    require_above_sma20 = st.checkbox("Close > SMA20", value=bool(profile_values["require_above_sma20"]))
    require_above_sma50 = st.checkbox("Close > SMA50", value=bool(profile_values["require_above_sma50"]))
    tsi323_min = st.slider("TSI(3,2,3) >", 50.0, 100.0, float(profile_values["tsi323_min"]), 1.0)
    tsi747_min = st.slider("TSI(7,4,7) >", 40.0, 100.0, float(profile_values["tsi747_min"]), 1.0)
    cci_min = st.slider("CCI(15) >", 50.0, 200.0, float(profile_values["cci_min"]), 1.0)
    require_cci_roll = st.checkbox("Today's CCI < yesterday's CCI", value=bool(profile_values["require_cci_roll"]))
    bbpct_min = st.slider("%B(20,2) >", 0.50, 1.50, float(profile_values["bbpct_min"]), 0.01)
    ext10_min = st.slider("Close / SMA10 - 1 >", 0.00, 0.10, float(profile_values["ext10_min"]), 0.005)
    top_n = st.slider("Top results", 5, 100, 25, 5)


def parse_universe() -> List[str]:
    symbols: List[str] = []

    if universe_mode == "Built-in live universe":
        symbols.extend(BUILTIN_UNIVERSES.get(preset_name, DEFAULT_UNIVERSE))

    if universe_mode == "Upload CSV/TXT" and uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                u = pd.read_csv(uploaded)
                if "ticker" in [c.lower() for c in u.columns]:
                    col = [c for c in u.columns if c.lower() == "ticker"][0]
                    symbols.extend(u[col].astype(str).tolist())
                else:
                    symbols.extend(u.iloc[:, 0].astype(str).tolist())
            else:
                symbols.extend(uploaded.read().decode("utf-8", errors="ignore").replace("\n", ",").split(","))
        except Exception:
            pass

    if universe_mode == "Paste tickers" and universe_text.strip():
        symbols.extend(universe_text.replace("\n", ",").split(","))

    cleaned: List[str] = []
    for s in symbols:
        t = sanitize_ticker(s)
        if t and t not in cleaned:
            cleaned.append(t)
    return cleaned or BUILTIN_UNIVERSES.get(preset_name, DEFAULT_UNIVERSE)

universe = parse_universe()

col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button("Refresh cache"):
        with st.spinner("Refreshing Yahoo cache..."):
            updated, failed = refresh_cache(universe, batch_size=batch_size, sleep_s=sleep_s, period=refresh_period)
        st.success(f"Cache refresh complete. Updated: {updated}, failed: {failed}")
with col2:
    if st.button("Scan now"):
        st.session_state["run_scan"] = True
with col3:
    active_logic = "Fire" if fire_mode else scan_profile_name
    st.caption(f"Universe size: {len(universe)} symbols. Cached files: {len(list(DAILY_CACHE.glob('*.parquet')))}")
    st.caption(f"Active scan logic: {active_logic}")

if st.session_state.get("run_scan", False):
    results = run_scan(
        universe=universe,
        min_price=min_price,
        min_avg_vol=min_avg_vol,
        tsi323_min=tsi323_min,
        tsi747_min=tsi747_min,
        cci_min=cci_min,
        bbpct_min=bbpct_min,
        ext10_min=ext10_min,
        require_above_sma20=require_above_sma20,
        require_above_sma50=require_above_sma50,
        require_cci_roll=require_cci_roll,
    )

    if results.empty:
        st.warning("No results matched the current diamond conditions. Refresh cache or loosen thresholds.")
    else:
        st.subheader("Diamond scan results")
        display = results.head(top_n).copy()
        rename_map = {
            "Adj Close": "Close",
            "AVG_VOL20": "AvgVol20",
            "CCI15_DELTA": "CCIΔ",
            "BBPCT": "%B",
            "EXT10": "Ext10",
            "HEAT_SCORE": "HeatScore",
            "DIAMOND_SCORE": "DiamondScore",
            "ULTIMATE_SCORE": "UltimateScore",
            "TSI_323_SCORE": "TSI323Score",
            "TSI_747_SCORE": "TSI747Score",
            "CCI15_SCORE": "CCIScore",
            "BBPCT_SCORE": "BBScore",
            "EXT10_SCORE": "ExtScore",
        }
        display = display.rename(columns=rename_map)
        st.dataframe(
            display.style.format({
                "Close":"{:.2f}","AvgVol20":"{:,.0f}","TSI_323":"{:.1f}","TSI_747":"{:.1f}","CCI15":"{:.1f}",
                "CCIΔ":"{:.1f}","%B":"{:.2f}","Ext10":"{:.3f}","HeatScore":"{:.1f}","DiamondScore":"{:.1f}",
                "UltimateScore":"{:.1f}","TSI323Score":"{:.1f}","TSI747Score":"{:.1f}","CCIScore":"{:.1f}","BBScore":"{:.1f}","ExtScore":"{:.1f}"
            }),
            use_container_width=True,
            hide_index=True,
        )

        snapshot_file = SNAPSHOT_CACHE / f"diamond_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results.to_csv(snapshot_file, index=False)
        st.download_button("Download scan snapshot", data=results.to_csv(index=False).encode("utf-8"), file_name=snapshot_file.name, mime="text/csv")

        selected = st.selectbox("Analyze ticker", display["Ticker"].tolist(), index=0)
        raw = load_cached_symbol(selected)
        feat = compute_features(raw)
        if not feat.empty:
            st.subheader(f"{selected} diamond analysis")
            last = feat.dropna(how="all").iloc[-1]
            prev = feat.dropna(how="all").iloc[-2] if len(feat.dropna(how="all")) > 1 else pd.Series(dtype=float)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Diamond Score", f"{safe_float(last.get('DIAMOND_SCORE')):.1f}", f"{safe_float(last.get('DIAMOND_SCORE') - prev.get('DIAMOND_SCORE', np.nan)):.1f}" if not prev.empty else None)
            m2.metric("Ultimate Score", f"{safe_float(last.get('ULTIMATE_SCORE')):.1f}")
            m3.metric("TSI(3,2,3)", f"{safe_float(last.get('TSI_323')):.1f}")
            m4.metric("CCI(15)", f"{safe_float(last.get('CCI15')):.1f}")

            st.plotly_chart(price_panel(feat.tail(252), selected), use_container_width=True)

            b1, b2 = st.columns(2)
            with b1:
                st.plotly_chart(empirical_bell_figure(feat["ULTIMATE_SCORE"], safe_float(last.get("ULTIMATE_SCORE")), safe_float(prev.get("ULTIMATE_SCORE", np.nan)), f"{selected} Ultimate Bell"), use_container_width=True)
                st.plotly_chart(empirical_bell_figure(feat["TSI_323_SCORE"], safe_float(last.get("TSI_323_SCORE")), safe_float(prev.get("TSI_323_SCORE", np.nan)), f"{selected} TSI(3,2,3) Bell"), use_container_width=True)
            with b2:
                st.plotly_chart(empirical_bell_figure(feat["CCI15_SCORE"], safe_float(last.get("CCI15_SCORE")), safe_float(prev.get("CCI15_SCORE", np.nan)), f"{selected} CCI(15) Bell"), use_container_width=True)
                st.plotly_chart(empirical_bell_figure(feat["BBPCT_SCORE"], safe_float(last.get("BBPCT_SCORE")), safe_float(prev.get("BBPCT_SCORE", np.nan)), f"{selected} %B Bell"), use_container_width=True)

            comp = pd.DataFrame({
                "Component": ["HeatScore", "DiamondScore", "UltimateScore", "TSI323Score", "TSI747Score", "CCIScore", "BBScore", "ExtScore"],
                "Value": [
                    safe_float(last.get("HEAT_SCORE")), safe_float(last.get("DIAMOND_SCORE")), safe_float(last.get("ULTIMATE_SCORE")),
                    safe_float(last.get("TSI_323_SCORE")), safe_float(last.get("TSI_747_SCORE")), safe_float(last.get("CCI15_SCORE")),
                    safe_float(last.get("BBPCT_SCORE")), safe_float(last.get("EXT10_SCORE"))
                ]
            })
            st.dataframe(comp.style.format({"Value":"{:.1f}"}), use_container_width=True, hide_index=True)
else:
    st.info("Using the built-in live universe by default. Pick a scan preset or enable Fire mode, then click Refresh cache and Scan now.")
