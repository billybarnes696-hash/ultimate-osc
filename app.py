import os
import io
import math
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from scipy.stats import gaussian_kde, norm

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Diamond Scanner", layout="wide")
st.title("💎 Diamond Scanner")
st.caption(
    "StockCharts-style overheated rollover scanner with persistent Yahoo cache, "
    "Gaussian/CDF bells, and an ultimate fade oscillator."
)

APP_TZ = timezone.utc
TODAY_UTC = datetime.now(APP_TZ).date()
CACHE_ROOT = Path("cache")
PRICE_CACHE_DIR = CACHE_ROOT / "prices_daily"
SNAPSHOT_CACHE_DIR = CACHE_ROOT / "snapshots"
META_CACHE_DIR = CACHE_ROOT / "meta"
for p in [PRICE_CACHE_DIR, SNAPSHOT_CACHE_DIR, META_CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_ETFS = [
    "SPY", "RSP", "QQQ", "IWM", "DIA", "SMH", "XLF", "XLK", "XLE", "XLI",
    "XLP", "XLY", "XLV", "XLC", "XLB", "XLU", "XBI", "ARKK", "HYG", "TLT",
    "IEF", "KRE", "XHB", "SOXX", "IGV", "IYT", "ITB", "XRT", "GDX", "SLV",
]

DEFAULT_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX", "AVGO",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "KKR", "SCHW", "COF",
    "GS", "BX", "APO", "CME", "ICE", "V", "MA", "PYPL", "SQ", "HOOD",
    "LLY", "UNH", "PFE", "MRK", "ABBV", "ISRG", "ABT", "TMO", "BSX", "SYK",
    "HD", "LOW", "AMZN", "TJX", "BKNG", "RCL", "CCL", "NKE", "LEVI", "DECK",
    "CAT", "DE", "PH", "ETN", "GE", "GEV", "FTV", "ITT", "EMR", "ROK",
    "INTC", "MU", "QCOM", "ANET", "PANW", "CRWD", "SNOW", "PLTR", "ORCL", "ADBE",
    "EXTR", "SHOP", "MELI", "UBER", "DASH", "ABNB", "CRM", "NOW", "MDB", "DDOG",
]

BUILTIN_UNIVERSE = sorted(set(DEFAULT_ETFS + DEFAULT_STOCKS))

SECTOR_MAP = {
    "XLF": "Financials ETF", "KRE": "Regional Banks ETF", "XLK": "Technology ETF", "SMH": "Semiconductors ETF",
    "XLE": "Energy ETF", "XLI": "Industrials ETF", "XLY": "Consumer Discretionary ETF", "XLP": "Consumer Staples ETF",
    "XLV": "Health Care ETF", "XLC": "Communication Services ETF", "XLB": "Materials ETF", "XLU": "Utilities ETF",
}

# =============================================================================
# HELPERS
# =============================================================================
def normalize_ticker(x: str) -> str:
    return str(x).strip().upper().replace(" ", "")


def parse_tickers_from_text(raw: str) -> List[str]:
    raw = raw.replace("\n", ",").replace(";", ",").replace("\t", ",")
    return [normalize_ticker(x) for x in raw.split(",") if normalize_ticker(x)]


def is_market_day(ts: pd.Timestamp) -> bool:
    return ts.dayofweek < 5


def today_str() -> str:
    return TODAY_UTC.strftime("%Y-%m-%d")


def parquet_path_for_symbol(symbol: str) -> Path:
    return PRICE_CACHE_DIR / f"{symbol}.parquet"


def meta_path_for_symbol(symbol: str) -> Path:
    return META_CACHE_DIR / f"{symbol}.json"


def load_symbol_cache(symbol: str) -> pd.DataFrame:
    path = parquet_path_for_symbol(symbol)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except Exception:
        return pd.DataFrame()


def save_symbol_cache(symbol: str, df: pd.DataFrame, source_note: str = "yahoo") -> None:
    if df.empty:
        return
    df = df.sort_index()
    df.to_parquet(parquet_path_for_symbol(symbol))
    meta = {
        "symbol": symbol,
        "last_write_utc": datetime.now(APP_TZ).isoformat(),
        "last_bar": str(df.index.max().date()),
        "rows": int(len(df)),
        "source": source_note,
    }
    meta_path_for_symbol(symbol).write_text(json.dumps(meta, indent=2))


def read_symbol_meta(symbol: str) -> Dict:
    path = meta_path_for_symbol(symbol)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def cache_status_for_symbol(symbol: str) -> str:
    meta = read_symbol_meta(symbol)
    if not meta:
        return "missing"
    try:
        last_bar = pd.Timestamp(meta.get("last_bar"))
    except Exception:
        return "stale"
    if last_bar.date() >= TODAY_UTC - timedelta(days=1):
        return "fresh"
    return "stale"


def chunked(seq: List[str], size: int) -> List[List[str]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def standardize_download(data: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if data is None or len(data) == 0:
        return out

    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = list(data.columns.get_level_values(0).unique())
        lvl1 = list(data.columns.get_level_values(1).unique())

        if set(symbols).intersection(lvl0):
            for sym in symbols:
                if sym in lvl0:
                    sub = data[sym].copy()
                    sub.columns = [str(c).title() for c in sub.columns]
                    out[sym] = sub
        elif set(symbols).intersection(lvl1):
            for sym in symbols:
                if sym in lvl1:
                    sub = data.xs(sym, axis=1, level=1).copy()
                    sub.columns = [str(c).title() for c in sub.columns]
                    out[sym] = sub
    else:
        # Single symbol
        sym = symbols[0]
        sub = data.copy()
        sub.columns = [str(c).title() for c in sub.columns]
        out[sym] = sub

    for sym, sub in list(out.items()):
        if sub.empty:
            continue
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in sub.columns]
        sub = sub[keep].copy()
        sub.index = pd.to_datetime(sub.index).tz_localize(None)
        sub = sub[~sub.index.duplicated(keep="last")].sort_index()
        out[sym] = sub
    return out


def download_batch(symbols: List[str], period: str = "2y", interval: str = "1d", start: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    kwargs = dict(
        tickers=symbols,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
        prepost=False,
    )
    if start:
        kwargs["start"] = start
    else:
        kwargs["period"] = period

    data = yf.download(**kwargs)
    return standardize_download(data, symbols)


def refresh_symbol_cache(symbols: List[str], full_refresh: bool, batch_size: int = 25, pause_seconds: float = 1.0) -> Tuple[List[str], List[str]]:
    refreshed, failed = [], []
    targets = [normalize_ticker(s) for s in symbols if normalize_ticker(s)]
    for batch in chunked(targets, batch_size):
        batch_start = None
        if not full_refresh:
            # use recent incremental window that still covers indicator lookbacks safely
            batch_start = (TODAY_UTC - timedelta(days=120)).strftime("%Y-%m-%d")
        try:
            pulled = download_batch(batch, period="2y", interval="1d", start=batch_start)
        except Exception:
            pulled = {}

        for sym in batch:
            new_df = pulled.get(sym, pd.DataFrame())
            old_df = load_symbol_cache(sym)
            if new_df.empty and old_df.empty:
                failed.append(sym)
                continue
            if old_df.empty:
                merged = new_df.copy()
            elif new_df.empty:
                merged = old_df.copy()
            else:
                merged = pd.concat([old_df, new_df], axis=0)
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            if merged.empty:
                failed.append(sym)
                continue
            save_symbol_cache(sym, merged, source_note="yahoo_incremental" if not full_refresh else "yahoo_full")
            refreshed.append(sym)
        time.sleep(pause_seconds)
    return refreshed, failed


def load_history_for_universe(symbols: List[str], min_bars: int = 120) -> Dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        df = load_symbol_cache(sym)
        if len(df) >= min_bars and "Close" in df.columns:
            out[sym] = df.copy()
    return out


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def tsi(close: pd.Series, long: int, short: int, signal: int) -> Tuple[pd.Series, pd.Series]:
    mtm = close.diff()
    abs_mtm = mtm.abs()
    tsi_raw = 100 * (ema(ema(mtm, long), short) / ema(ema(abs_mtm, long), short).replace(0, np.nan))
    sig = ema(tsi_raw, signal)
    return tsi_raw, sig


def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def bb_percent_b(close: pd.Series, length: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return (close - lower) / (upper - lower).replace(0, np.nan)


def robust_zscore(series: pd.Series, lookback: int = 126) -> pd.Series:
    med = series.rolling(lookback).median()
    mad = series.rolling(lookback).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    robust_std = 1.4826 * mad.replace(0, np.nan)
    return (series - med) / robust_std


def rolling_percentile(series: pd.Series, lookback: int = 126) -> pd.Series:
    def pct_rank(x: np.ndarray) -> float:
        valid = x[~np.isnan(x)]
        if len(valid) < max(20, int(lookback * 0.5)):
            return np.nan
        return 100.0 * (valid <= valid[-1]).sum() / len(valid)
    return series.rolling(lookback, min_periods=max(20, int(lookback * 0.5))).apply(pct_rank, raw=True)


def gaussian_cdf_score(series: pd.Series, lookback: int = 126, clip_z: float = 3.5) -> pd.Series:
    z = robust_zscore(series, lookback=lookback).clip(-clip_z, clip_z)
    return pd.Series(norm.cdf(z) * 100.0, index=series.index)


def normalized_composite(values: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    cols = [k for k in weights.keys() if k in values]
    if not cols:
        return pd.Series(dtype=float)
    df = pd.concat([values[c] for c in cols], axis=1)
    df.columns = cols
    w = pd.Series({c: weights[c] for c in cols}, dtype=float)
    numerator = df.mul(w, axis=1).sum(axis=1, skipna=True)
    denom = df.notna().mul(w, axis=1).sum(axis=1).replace(0, np.nan)
    return numerator / denom


def build_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df.get("High", close), errors="coerce")
    low = pd.to_numeric(df.get("Low", close), errors="coerce")
    volume = pd.to_numeric(df.get("Volume"), errors="coerce")

    out = pd.DataFrame(index=df.index)
    out["Close"] = close
    out["Volume"] = volume
    out["SMA10"] = close.rolling(10).mean()
    out["SMA20"] = close.rolling(20).mean()
    out["SMA50"] = close.rolling(50).mean()
    out["AVG_VOL20"] = volume.rolling(20).mean()
    out["EXT10"] = close / out["SMA10"] - 1.0
    out["BBPCT"] = bb_percent_b(close, 20, 2.0)
    out["CCI15"] = cci(high, low, close, 15)
    out["CCI15_DELTA"] = out["CCI15"].diff()

    tsi_323, tsi_323_sig = tsi(close, 3, 2, 3)
    tsi_747, tsi_747_sig = tsi(close, 7, 4, 7)
    out["TSI_323"] = tsi_323
    out["TSI_323_SIG"] = tsi_323_sig
    out["TSI_747"] = tsi_747
    out["TSI_747_SIG"] = tsi_747_sig

    # institutional-style bell ingredients
    out["TSI_323_PCT"] = rolling_percentile(out["TSI_323"], 126)
    out["TSI_747_PCT"] = rolling_percentile(out["TSI_747"], 126)
    out["CCI15_PCT"] = rolling_percentile(out["CCI15"], 126)
    out["BBPCT_PCT"] = rolling_percentile(out["BBPCT"], 126)
    out["EXT10_PCT"] = rolling_percentile(out["EXT10"], 126)

    bell_inputs = {
        "TSI_323_BELL": gaussian_cdf_score(out["TSI_323"], 126),
        "TSI_747_BELL": gaussian_cdf_score(out["TSI_747"], 126),
        "CCI15_BELL": gaussian_cdf_score(out["CCI15"], 126),
        "BBPCT_BELL": gaussian_cdf_score(out["BBPCT"], 126),
        "EXT10_BELL": gaussian_cdf_score(out["EXT10"], 126),
    }
    for k, v in bell_inputs.items():
        out[k] = v

    comp_weights = {
        "TSI_323_BELL": 0.28,
        "TSI_747_BELL": 0.20,
        "CCI15_BELL": 0.22,
        "BBPCT_BELL": 0.18,
        "EXT10_BELL": 0.12,
    }
    out["ULTIMATE_BELL"] = normalized_composite({k: out[k] for k in comp_weights}, comp_weights)
    out["ULTIMATE_SLOPE3"] = out["ULTIMATE_BELL"].diff(3)
    out["FREEFALL_SCORE"] = (
        0.35 * out["ULTIMATE_BELL"] +
        0.20 * out["TSI_323_PCT"] +
        0.15 * out["CCI15_PCT"] +
        0.15 * out["BBPCT_PCT"] +
        0.15 * out["EXT10_PCT"]
    )
    return out


def sector_for_symbol(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "Unknown")


def scan_latest(history_map: Dict[str, pd.DataFrame], require_proxy_optionable: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows = []
    indicator_map: Dict[str, pd.DataFrame] = {}

    for sym, hist in history_map.items():
        try:
            ind = build_indicator_frame(hist)
        except Exception:
            continue
        indicator_map[sym] = ind
        if len(ind.dropna()) < 60:
            continue
        last = ind.iloc[-1]
        prev = ind.iloc[-2] if len(ind) > 1 else last

        optionable_proxy = last["AVG_VOL20"] >= 1_000_000 and last["Close"] > 5
        if require_proxy_optionable and not optionable_proxy:
            continue

        pass_scan = (
            (last["Close"] > 5) and
            (last["AVG_VOL20"] > 1_000_000) and
            (last["Close"] > last["SMA20"]) and
            (last["Close"] > last["SMA50"]) and
            (last["TSI_323"] > 95) and
            (last["TSI_747"] > 70) and
            (last["CCI15"] > 90) and
            (last["CCI15"] < prev["CCI15"]) and
            (last["BBPCT"] > 0.95) and
            (last["EXT10"] > 0.03)
        )

        score = (
            22 * min(1.2, max(0, last["TSI_323"] / 100.0)) +
            13 * min(1.2, max(0, last["TSI_747"] / 80.0)) +
            15 * min(1.2, max(0, last["CCI15"] / 120.0)) +
            15 * min(1.2, max(0, last["BBPCT"] / 1.05)) +
            12 * min(1.2, max(0, last["EXT10"] / 0.05)) +
            10 * min(1.2, max(0, last["ULTIMATE_BELL"] / 100.0)) +
            13 * (1 if last["ULTIMATE_SLOPE3"] < 0 else 0)
        )

        rows.append({
            "Ticker": sym,
            "Date": ind.index[-1].date(),
            "Sector": sector_for_symbol(sym),
            "Close": last["Close"],
            "AvgVol20": last["AVG_VOL20"],
            "TSI(3,2,3)": last["TSI_323"],
            "TSI(7,4,7)": last["TSI_747"],
            "CCI(15)": last["CCI15"],
            "CCIΔ": last["CCI15"] - prev["CCI15"],
            "%B(20,2)": last["BBPCT"],
            "Ext10": last["EXT10"],
            "UltimateBell": last["ULTIMATE_BELL"],
            "FreeFallScore": last["FREEFALL_SCORE"],
            "DiamondScore": score,
            "ScanPass": pass_scan,
            "ProxyOptionable": bool(optionable_proxy),
            "Freshness": cache_status_for_symbol(sym),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res, indicator_map
    res = res.sort_values(["ScanPass", "DiamondScore", "%B(20,2)"], ascending=[False, False, False]).reset_index(drop=True)
    return res, indicator_map


def bell_curve_figure(series: pd.Series, current_val: float, title: str, x_label: str = "Score") -> go.Figure:
    s = pd.to_numeric(series, errors="coerce").dropna().tail(252)
    fig = go.Figure()
    x = np.linspace(0, 100, 250)
    if len(s) >= 25:
        try:
            kde = gaussian_kde(s.values)
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

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy", name="Density"))
    for lo, hi, color in [(0, 20, "rgba(0,200,83,0.10)"), (20, 40, "rgba(144,238,144,0.10)"), (40, 60, "rgba(255,245,157,0.10)"), (60, 80, "rgba(255,224,130,0.10)"), (80, 100, "rgba(255,23,68,0.10)")]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, line_width=0)
    if pd.notna(current_val):
        yv = np.interp(current_val, x, y)
        fig.add_trace(go.Scatter(
            x=[current_val], y=[yv], mode="markers+text", text=[f"{current_val:.1f}"], textposition="top center",
            marker=dict(size=14, line=dict(width=2, color="black"))
        ))
    fig.update_layout(title=title, height=240, margin=dict(l=10, r=10, t=45, b=10), xaxis_title=x_label, yaxis_visible=False)
    return fig


def line_stack_figure(ind: pd.DataFrame, symbol: str) -> go.Figure:
    show = ind.tail(180).copy()
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=[f"{symbol} Price", "Ultimate Bell", "TSI / CCI", "%B / Ext10", "Diamond Score"])
    fig.add_trace(go.Scatter(x=show.index, y=show["Close"], name="Close"), row=1, col=1)
    if "SMA10" in show:
        fig.add_trace(go.Scatter(x=show.index, y=show["SMA10"], name="SMA10"), row=1, col=1)
    if "SMA20" in show:
        fig.add_trace(go.Scatter(x=show.index, y=show["SMA20"], name="SMA20"), row=1, col=1)

    fig.add_trace(go.Scatter(x=show.index, y=show["ULTIMATE_BELL"], name="UltimateBell"), row=2, col=1)
    fig.add_hline(y=80, row=2, col=1, line_dash="dot")
    fig.add_hline(y=50, row=2, col=1, line_dash="dash")

    fig.add_trace(go.Scatter(x=show.index, y=show["TSI_323"], name="TSI323"), row=3, col=1)
    fig.add_trace(go.Scatter(x=show.index, y=show["TSI_747"], name="TSI747"), row=3, col=1)
    fig.add_trace(go.Scatter(x=show.index, y=show["CCI15"], name="CCI15"), row=3, col=1)
    fig.add_hline(y=100, row=3, col=1, line_dash="dot")
    fig.add_hline(y=90, row=3, col=1, line_dash="dot")

    fig.add_trace(go.Scatter(x=show.index, y=show["BBPCT"], name="%B"), row=4, col=1)
    fig.add_trace(go.Scatter(x=show.index, y=show["EXT10"] * 100, name="Ext10 %"), row=4, col=1)
    fig.add_hline(y=0.95, row=4, col=1, line_dash="dot")

    fig.add_trace(go.Scatter(x=show.index, y=show["DiamondScore"], name="DiamondScore") if "DiamondScore" in show.columns else go.Scatter(x=show.index, y=show["FREEFALL_SCORE"], name="FreeFallScore"), row=5, col=1)
    fig.update_layout(height=1100, margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation="h"))
    return fig


def build_symbol_story(ind: pd.DataFrame) -> pd.DataFrame:
    out = ind.copy()
    out["DiamondScore"] = (
        22 * np.minimum(1.2, np.maximum(0, out["TSI_323"] / 100.0)) +
        13 * np.minimum(1.2, np.maximum(0, out["TSI_747"] / 80.0)) +
        15 * np.minimum(1.2, np.maximum(0, out["CCI15"] / 120.0)) +
        15 * np.minimum(1.2, np.maximum(0, out["BBPCT"] / 1.05)) +
        12 * np.minimum(1.2, np.maximum(0, out["EXT10"] / 0.05)) +
        10 * np.minimum(1.2, np.maximum(0, out["ULTIMATE_BELL"] / 100.0)) +
        13 * (out["ULTIMATE_SLOPE3"] < 0).astype(int)
    )
    return out


# =============================================================================
# UI
# =============================================================================
with st.sidebar:
    st.header("Universe")
    use_builtin = st.checkbox("Use built-in liquid stock + ETF universe", value=True)
    custom_text = st.text_area("Add tickers (comma or newline separated)", value="")
    upload_csv = st.file_uploader("Optional CSV with a 'ticker' column", type=["csv"])

    st.header("Yahoo cache")
    full_refresh = st.checkbox("Force full refresh (slower)", value=False)
    refresh_requested = st.button("Refresh cached data")
    batch_size = st.slider("Batch size", 10, 50, 25, 5)
    pause_seconds = st.slider("Pause between batches (sec)", 0.0, 3.0, 1.0, 0.25)

    st.header("Scan")
    show_only_pass = st.checkbox("Show only exact scan passes", value=True)
    require_proxy_optionable = st.checkbox("Require optionable proxy", value=True,
                                           help="Yahoo does not reliably expose optionability in bulk. This proxy uses price > 5 and AvgVol20 > 1M.")
    top_n = st.slider("Top rows", 10, 100, 30, 5)

    st.header("Notes")
    st.caption(
        "This app avoids Yahoo rerun abuse by using persistent per-symbol parquet files. "
        "Refresh updates the local warehouse; scans then run locally and return in seconds."
    )

# Build universe
universe: List[str] = []
if use_builtin:
    universe.extend(BUILTIN_UNIVERSE)
if custom_text.strip():
    universe.extend(parse_tickers_from_text(custom_text))
if upload_csv is not None:
    try:
        df_up = pd.read_csv(upload_csv)
        ticker_col = next((c for c in df_up.columns if str(c).lower().strip() in {"ticker", "symbol", "tickers"}), None)
        if ticker_col:
            universe.extend([normalize_ticker(x) for x in df_up[ticker_col].dropna().tolist()])
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded CSV: {e}")
universe = sorted(set([u for u in universe if u]))

st.markdown("### Universe")
st.write(f"Loaded **{len(universe)}** tickers.")

if refresh_requested and universe:
    with st.spinner("Refreshing local Yahoo cache..."):
        refreshed, failed = refresh_symbol_cache(universe, full_refresh=full_refresh, batch_size=batch_size, pause_seconds=pause_seconds)
    st.success(f"Refreshed {len(refreshed)} symbols.")
    if failed:
        st.warning(f"Failed or missing: {', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}")

history_map = load_history_for_universe(universe, min_bars=120)
cache_counts = pd.Series([cache_status_for_symbol(s) for s in universe]).value_counts().to_dict() if universe else {}
col1, col2, col3 = st.columns(3)
col1.metric("Cached symbols", len(history_map))
col2.metric("Fresh", cache_counts.get("fresh", 0))
col3.metric("Stale or missing", cache_counts.get("stale", 0) + cache_counts.get("missing", 0))

if not universe:
    st.info("Add tickers in the sidebar to build the scanner universe.")
    st.stop()

if len(history_map) == 0:
    st.warning("No cached histories yet. Click 'Refresh cached data' first.")
    st.stop()

with st.spinner("Running local scanner from cached warehouse..."):
    scan_df, indicator_map = scan_latest(history_map, require_proxy_optionable=require_proxy_optionable)

if scan_df.empty:
    st.warning("No scannable symbols found from cache.")
    st.stop()

# Filter results
result_df = scan_df.copy()
if show_only_pass:
    result_df = result_df[result_df["ScanPass"]].copy()
result_df = result_df.head(top_n)

st.markdown("### Scan results")
st.caption(
    "Exact scan conditions: close > 5, avg vol20 > 1M, close > SMA20, close > SMA50, "
    "TSI(3,2,3) > 95, TSI(7,4,7) > 70, CCI(15) > 90, today's CCI < yesterday's CCI, %B > 0.95, Ext10 > 3%."
)

styled = result_df[[
    "Ticker", "Sector", "Date", "Close", "TSI(3,2,3)", "TSI(7,4,7)", "CCI(15)", "CCIΔ", "%B(20,2)", "Ext10", "UltimateBell", "DiamondScore", "Freshness"
]].copy()
st.dataframe(
    styled.style.format({
        "Close": "{:.2f}", "TSI(3,2,3)": "{:.1f}", "TSI(7,4,7)": "{:.1f}", "CCI(15)": "{:.1f}",
        "CCIΔ": "{:.1f}", "%B(20,2)": "{:.2f}", "Ext10": "{:.2%}", "UltimateBell": "{:.1f}", "DiamondScore": "{:.1f}"
    }),
    use_container_width=True,
    hide_index=True,
)

sector_counts = result_df["Sector"].value_counts().reset_index()
sector_counts.columns = ["Sector", "Count"]
left, right = st.columns([1, 2])
with left:
    st.markdown("#### Sector concentration")
    st.dataframe(sector_counts, use_container_width=True, hide_index=True)
with right:
    fig_sector = go.Figure(go.Bar(x=sector_counts["Sector"], y=sector_counts["Count"]))
    fig_sector.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(fig_sector, use_container_width=True)

# Ticker detail
available_detail = result_df["Ticker"].tolist() if len(result_df) else scan_df["Ticker"].tolist()
default_pick = available_detail[0] if available_detail else scan_df.iloc[0]["Ticker"]
selected_ticker = st.selectbox("Analyze ticker", options=scan_df["Ticker"].tolist(), index=scan_df["Ticker"].tolist().index(default_pick) if default_pick in scan_df["Ticker"].tolist() else 0)

ind = indicator_map[selected_ticker].copy()
ind = build_symbol_story(ind)
last = ind.iloc[-1]
prev = ind.iloc[-2] if len(ind) > 1 else last

st.markdown(f"### {selected_ticker} diamond analysis")
metrics = st.columns(6)
metrics[0].metric("Close", f"{last['Close']:.2f}")
metrics[1].metric("DiamondScore", f"{last['DiamondScore']:.1f}")
metrics[2].metric("UltimateBell", f"{last['ULTIMATE_BELL']:.1f}", f"{(last['ULTIMATE_BELL'] - prev['ULTIMATE_BELL']):.1f}")
metrics[3].metric("TSI(3,2,3)", f"{last['TSI_323']:.1f}", f"{(last['TSI_323'] - prev['TSI_323']):.1f}")
metrics[4].metric("CCI(15)", f"{last['CCI15']:.1f}", f"{(last['CCI15'] - prev['CCI15']):.1f}")
metrics[5].metric("%B(20,2)", f"{last['BBPCT']:.2f}", f"{(last['BBPCT'] - prev['BBPCT']):.2f}")

st.plotly_chart(line_stack_figure(ind, selected_ticker), use_container_width=True)

b1, b2 = st.columns(2)
with b1:
    st.plotly_chart(bell_curve_figure(ind["ULTIMATE_BELL"], float(last["ULTIMATE_BELL"]), "Ultimate Bell Distribution"), use_container_width=True)
    st.plotly_chart(bell_curve_figure(ind["TSI_323_BELL"], float(last["TSI_323_BELL"]), "TSI(3,2,3) Bell"), use_container_width=True)
with b2:
    st.plotly_chart(bell_curve_figure(ind["CCI15_BELL"], float(last["CCI15_BELL"]), "CCI(15) Bell"), use_container_width=True)
    st.plotly_chart(bell_curve_figure(ind["BBPCT_BELL"], float(last["BBPCT_BELL"]), "%B Bell"), use_container_width=True)

with st.expander("Latest indicator snapshot"):
    latest_cols = [
        "Close", "AVG_VOL20", "SMA10", "SMA20", "SMA50", "TSI_323", "TSI_747", "CCI15", "BBPCT", "EXT10",
        "TSI_323_BELL", "TSI_747_BELL", "CCI15_BELL", "BBPCT_BELL", "EXT10_BELL", "ULTIMATE_BELL", "FREEFALL_SCORE", "DiamondScore"
    ]
    snap = ind[latest_cols].tail(10).copy()
    st.dataframe(snap.style.format({c: "{:.2f}" for c in snap.columns}), use_container_width=True)

# download snapshot cache summary
snapshot_df = scan_df.copy()
snapshot_file = SNAPSHOT_CACHE_DIR / f"diamond_scan_{today_str()}.csv"
snapshot_df.to_csv(snapshot_file, index=False)
with st.expander("Download today's local scan snapshot"):
    st.download_button(
        "Download scan CSV",
        data=snapshot_df.to_csv(index=False).encode("utf-8"),
        file_name=snapshot_file.name,
        mime="text/csv",
    )
