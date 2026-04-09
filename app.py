import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import gaussian_kde, norm

# =============================================================================
# PAGE
# =============================================================================
st.set_page_config(page_title="Diamond Scanner", layout="wide")
st.title("💎 Diamond Scanner")
st.caption(
    "Warehouse-first scanner: refresh Yahoo into local cache, scan locally in seconds, "
    "and grade any stock on the diamond scale."
)

# =============================================================================
# STORAGE
# =============================================================================
CACHE_ROOT = Path("cache")
DAILY_CACHE = CACHE_ROOT / "daily"
INDICATOR_CACHE = CACHE_ROOT / "indicator_rows"
SNAPSHOT_CACHE = CACHE_ROOT / "snapshots"
for p in [CACHE_ROOT, DAILY_CACHE, INDICATOR_CACHE, SNAPSHOT_CACHE]:
    p.mkdir(parents=True, exist_ok=True)

INDICATOR_SNAPSHOT_FILE = INDICATOR_CACHE / "latest_indicator_snapshot.parquet"
REFRESH_STATUS_FILE = INDICATOR_CACHE / "refresh_status.parquet"

# =============================================================================
# UNIVERSE
# =============================================================================
ETF_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "RSP", "SMH", "XLF", "XLK", "XLE", "XLI", "XLP", "XLV",
    "XLY", "XLC", "XLB", "XLRE", "XBI", "ARKK", "SOXX", "KRE", "IBB", "IGV", "GDX", "TLT",
    "HYG", "IYT", "XHB", "XRT", "XOP", "FDN", "ITA", "KBE", "KIE"
]

STOCK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX", "AVGO", "COST",
    "JPM", "GS", "MS", "BAC", "WFC", "C", "SCHW", "BLK", "UNH", "LLY", "ABBV", "MRK",
    "ISRG", "VRTX", "ABT", "TMO", "DHR", "HD", "LOW", "MCD", "SBUX", "BKNG", "UBER",
    "ORLY", "TJX", "CAT", "DE", "GE", "ETN", "PH", "HON", "MMM", "UNP", "CSX", "NSC",
    "LMT", "RTX", "BA", "NOC", "INTC", "MU", "QCOM", "AMAT", "LRCX", "KLAC", "CRM",
    "ORCL", "ADBE", "NOW", "PANW", "CRWD", "PLTR", "SNOW", "SHOP", "MDB", "ANET", "V",
    "MA", "PYPL", "COIN", "HOOD", "AXP", "CB", "PGR", "LEVI", "EXTR", "FTV", "ITT",
    "FIS", "FI", "ICE", "CME", "KKR", "BX", "APO", "HWM", "TT", "URI", "DDOG", "SQ"
]

DEFAULT_UNIVERSE = sorted(set(ETF_UNIVERSE + STOCK_UNIVERSE))
ETF_SET = set(ETF_UNIVERSE)

BUILTIN_UNIVERSES = {
    "Default live universe": DEFAULT_UNIVERSE,
    "Starter 40": DEFAULT_UNIVERSE[:40],
    "ETFs only": sorted(ETF_SET),
    "Semis + AI": ["SMH", "SOXX", "NVDA", "AMD", "AVGO", "MU", "QCOM", "AMAT", "LRCX", "KLAC", "INTC", "ANET", "PLTR", "TSLA"],
    "Financials focus": ["XLF", "KRE", "JPM", "GS", "MS", "BAC", "WFC", "C", "SCHW", "BLK", "KKR", "BX", "APO", "ICE", "CME", "AXP", "CB", "PGR", "FIS", "FI", "HOOD", "COIN"],
}

# =============================================================================
# SCAN PROFILES
# =============================================================================
# Default watchlist = exactly the user’s requested StockCharts-style logic.
SCAN_PROFILES = {
    "Watchlist": {
        "min_price": 5.0,
        "min_avg_vol": 1_000_000.0,
        "require_above_sma20": True,
        "require_above_sma50": True,
        "require_optionable": True,
        "tsi323_min": 95.0,
        "tsi747_min": 70.0,
        "cci_min": 90.0,
        "require_cci_roll": True,
        "bbpct_min": 0.95,
        "ext10_min": 0.03,
        "rank_by": "%B",
    },
    "Standard Diamond": {
        "min_price": 5.0,
        "min_avg_vol": 1_000_000.0,
        "require_above_sma20": True,
        "require_above_sma50": True,
        "require_optionable": True,
        "tsi323_min": 95.0,
        "tsi747_min": 70.0,
        "cci_min": 90.0,
        "require_cci_roll": True,
        "bbpct_min": 0.95,
        "ext10_min": 0.03,
        "rank_by": "DiamondScore",
    },
    "Aggressive Fade": {
        "min_price": 5.0,
        "min_avg_vol": 1_000_000.0,
        "require_above_sma20": True,
        "require_above_sma50": True,
        "require_optionable": True,
        "tsi323_min": 96.0,
        "tsi747_min": 72.0,
        "cci_min": 95.0,
        "require_cci_roll": True,
        "bbpct_min": 0.98,
        "ext10_min": 0.032,
        "rank_by": "DiamondScore",
    },
    "Custom": None,
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
    "rank_by": "DiamondScore",
}

# =============================================================================
# HELPERS
# =============================================================================
def sanitize_ticker(t: str) -> str:
    return str(t).strip().upper().replace("/", "-")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def parquet_path(symbol: str) -> Path:
    return DAILY_CACHE / f"{sanitize_ticker(symbol)}.parquet"


def load_cached_symbol(symbol: str) -> pd.DataFrame:
    path = parquet_path(symbol)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def save_cached_symbol(symbol: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df = df.copy().sort_index()
    df.to_parquet(parquet_path(symbol))


def load_indicator_snapshot() -> pd.DataFrame:
    if not INDICATOR_SNAPSHOT_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(INDICATOR_SNAPSHOT_FILE)
    except Exception:
        return pd.DataFrame()


def save_indicator_snapshot(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.to_parquet(INDICATOR_SNAPSHOT_FILE, index=False)


def update_indicator_snapshot(rows: List[pd.Series]) -> None:
    if not rows:
        return
    existing = load_indicator_snapshot()
    new_df = pd.DataFrame(rows)
    if existing.empty:
        merged = new_df
    else:
        merged = pd.concat([existing, new_df], axis=0, ignore_index=True)
        merged = merged.sort_values(["Ticker", "Date"]).drop_duplicates(subset=["Ticker"], keep="last")
    save_indicator_snapshot(merged)


def save_refresh_status(status_rows: List[Dict]) -> None:
    if not status_rows:
        return
    new_df = pd.DataFrame(status_rows)
    if REFRESH_STATUS_FILE.exists():
        try:
            existing = pd.read_parquet(REFRESH_STATUS_FILE)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        except Exception:
            pass
    new_df = new_df.tail(500)
    new_df.to_parquet(REFRESH_STATUS_FILE, index=False)


def load_refresh_status() -> pd.DataFrame:
    if not REFRESH_STATUS_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(REFRESH_STATUS_FILE)
    except Exception:
        return pd.DataFrame()


def ticker_assumed_optionable(symbol: str) -> bool:
    # Warehouse-safe proxy to avoid expensive option-chain calls.
    # All curated built-in names are assumed optionable.
    return sanitize_ticker(symbol) in set(DEFAULT_UNIVERSE)


def apply_scan_profile(profile_name: str, fire_mode: bool = False) -> Dict[str, float]:
    base = SCAN_PROFILES.get(profile_name)
    if base is None:
        base = SCAN_PROFILES["Watchlist"].copy()
    else:
        base = base.copy()
    if fire_mode:
        for key, value in FIRE_OVERRIDES.items():
            if isinstance(value, bool):
                base[key] = value
            elif key == "rank_by":
                base[key] = value
            else:
                base[key] = max(float(base.get(key, value)), float(value))
    return base


def symbol_is_stale(df: pd.DataFrame, refresh_days: int = 1) -> bool:
    if df.empty:
        return True
    last = pd.to_datetime(df.index.max()).tz_localize(None)
    now = pd.Timestamp.now().tz_localize(None)
    if now.normalize() <= last.normalize():
        return False
    return (now.normalize() - last.normalize()).days >= refresh_days


def batch_download(symbols: List[str], period: str = "2y") -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    return yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )


def normalize_download(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    symbol = sanitize_ticker(symbol)
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol not in raw.columns.get_level_values(0):
                return pd.DataFrame()
            out = raw[symbol].copy()
        else:
            out = raw.copy()
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in out.columns]
        out = out[keep].copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        if "Adj Close" not in out.columns and "Close" in out.columns:
            out["Adj Close"] = out["Close"]
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(how="all")
        return out[~out.index.duplicated(keep="last")].sort_index()
    except Exception:
        return pd.DataFrame()


def refresh_one_symbol(symbol: str, period: str = "2y") -> Tuple[bool, str]:
    try:
        raw = batch_download([symbol], period=period)
        df = normalize_download(raw, symbol)
        if df.empty:
            return False, "empty"
        old = load_cached_symbol(symbol)
        if not old.empty:
            df = pd.concat([old, df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        save_cached_symbol(symbol, df)
        feat = compute_features(df)
        if feat.empty:
            return False, "feature-empty"
        update_indicator_snapshot([latest_row_with_symbol(symbol, feat)])
        return True, "ok"
    except Exception as e:
        return False, str(e)


def refresh_cache(symbols: List[str], batch_size: int, sleep_s: float, period: str, max_refresh_symbols: int) -> Tuple[int, int, int]:
    symbols = [sanitize_ticker(s) for s in symbols]
    stale = [s for s in symbols if symbol_is_stale(load_cached_symbol(s), refresh_days=1)]
    stale = stale[:max_refresh_symbols]

    updated = 0
    failed = 0
    skipped = max(0, len(symbols) - len(stale))
    status_rows = []
    latest_rows = []

    for i in range(0, len(stale), batch_size):
        batch = stale[i:i + batch_size]
        try:
            raw = batch_download(batch, period=period)
        except Exception as e:
            failed += len(batch)
            for sym in batch:
                status_rows.append({"ts": datetime.now(), "Ticker": sym, "status": f"batch-fail: {e}"})
            time.sleep(max(sleep_s, 0.5))
            continue

        for sym in batch:
            try:
                df = normalize_download(raw, sym)
                if df.empty:
                    failed += 1
                    status_rows.append({"ts": datetime.now(), "Ticker": sym, "status": "empty-download"})
                    continue
                old = load_cached_symbol(sym)
                if not old.empty:
                    df = pd.concat([old, df]).sort_index()
                    df = df[~df.index.duplicated(keep="last")]
                save_cached_symbol(sym, df)
                feat = compute_features(df)
                if feat.empty:
                    failed += 1
                    status_rows.append({"ts": datetime.now(), "Ticker": sym, "status": "feature-empty"})
                    continue
                latest_rows.append(latest_row_with_symbol(sym, feat))
                updated += 1
                status_rows.append({"ts": datetime.now(), "Ticker": sym, "status": "ok"})
            except Exception as e:
                failed += 1
                status_rows.append({"ts": datetime.now(), "Ticker": sym, "status": f"symbol-fail: {e}"})
        if sleep_s > 0:
            time.sleep(sleep_s)

    if latest_rows:
        update_indicator_snapshot(latest_rows)
    if status_rows:
        save_refresh_status(status_rows)
    return updated, failed, skipped

# =============================================================================
# INDICATORS
# =============================================================================
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
        return 100.0 * s.rank(pct=True).iloc[-1]
    return series.rolling(lookback, min_periods=max(20, lookback // 4)).apply(_pct, raw=False)


def robust_cdf(series: pd.Series, lookback: int = 252) -> pd.Series:
    min_periods = max(20, lookback // 4)
    med = series.rolling(lookback, min_periods=min_periods).median()

    def _mad(x):
        s = pd.Series(x).dropna()
        if len(s) < min_periods:
            return np.nan
        m = np.median(s)
        return float(np.median(np.abs(s - m)))

    mad = series.rolling(lookback, min_periods=min_periods).apply(_mad, raw=False)
    robust_std = 1.4826 * mad.replace(0, np.nan)
    z = ((series - med) / robust_std).clip(-3.5, 3.5)
    return pd.Series(norm.cdf(z) * 100.0, index=series.index)


def empirical_score(series: pd.Series, lookback: int = 252, invert: bool = False, smooth: int = 3) -> pd.Series:
    p = percentile_rank(series, lookback)
    c = robust_cdf(series, lookback)
    slope = series.diff().ewm(span=5, adjust=False).mean()
    persist = 50 + np.tanh(slope.fillna(0) * 5) * 12
    score = 0.50 * p + 0.35 * c + 0.15 * persist
    if invert:
        score = 100 - score
    if smooth and smooth > 1:
        score = score.ewm(span=smooth, adjust=False).mean()
    return score.clip(0, 100)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or len(df) < 60:
        return pd.DataFrame()

    out = df.copy().sort_index()
    px = out["Adj Close"].fillna(out["Close"]) if "Adj Close" in out.columns else out["Close"]

    out["PX"] = px
    out["SMA10"] = px.rolling(10).mean()
    out["SMA20"] = px.rolling(20).mean()
    out["SMA50"] = px.rolling(50).mean()
    out["AVG_VOL20"] = out["Volume"].rolling(20).mean()
    out["EXT10"] = px / out["SMA10"] - 1.0

    out["TSI_323"], out["TSI_323_SIG"] = tsi(px, 3, 2, 3)
    out["TSI_747"], out["TSI_747_SIG"] = tsi(px, 7, 4, 7)
    out["CCI15"] = cci(out, 15)
    out["CCI15_DELTA"] = out["CCI15"].diff()
    out["BBPCT"] = bb_pct(px, 20, 2.0)

    out["TSI_323_SCORE"] = empirical_score(out["TSI_323"], 252)
    out["TSI_747_SCORE"] = empirical_score(out["TSI_747"], 252)
    out["CCI15_SCORE"] = empirical_score(out["CCI15"], 252)
    out["BBPCT_SCORE"] = empirical_score(out["BBPCT"], 252)
    out["EXT10_SCORE"] = empirical_score(out["EXT10"], 252)

    cci_roll_component = (50 + np.tanh((-out["CCI15_DELTA"].fillna(0)) / 8.0) * 50).clip(0, 100)
    signal_gap = (out["TSI_323"] - out["TSI_323_SIG"]).fillna(0)
    gap_component = (50 + np.tanh(signal_gap / 6.0) * 20).clip(0, 100)
    overheat_penalty = np.where(out["TSI_323"] > 99.5, 8, 0) + np.where(out["BBPCT"] > 1.10, 8, 0)

    out["HEAT_SCORE"] = (
        0.33 * out["TSI_323_SCORE"] +
        0.23 * out["TSI_747_SCORE"] +
        0.22 * out["CCI15_SCORE"] +
        0.22 * out["BBPCT_SCORE"]
    )

    out["DIAMOND_SCORE"] = (
        0.45 * out["HEAT_SCORE"] +
        0.25 * cci_roll_component +
        0.20 * out["EXT10_SCORE"] +
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

    out["MEETS_WATCHLIST"] = (
        (px > 5) &
        (out["AVG_VOL20"] > 1_000_000) &
        (px > out["SMA20"]) &
        (px > out["SMA50"]) &
        (out["TSI_323"] > 95) &
        (out["TSI_747"] > 70) &
        (out["CCI15"] > 90) &
        (out["CCI15_DELTA"] < 0) &
        (out["BBPCT"] > 0.95) &
        (out["EXT10"] > 0.03)
    )
    return out


def latest_row_with_symbol(symbol: str, df: pd.DataFrame) -> pd.Series:
    base = df.dropna(how="all")
    row = base.iloc[-1].copy()
    row["Ticker"] = sanitize_ticker(symbol)
    row["Date"] = base.index[-1]
    row["Close"] = safe_float(row.get("Adj Close", row.get("PX", row.get("Close", np.nan))))
    row["Group"] = "ETF" if sanitize_ticker(symbol) in ETF_SET else "Stock"
    row["OptionableAssumed"] = ticker_assumed_optionable(symbol)
    row["Fresh"] = not symbol_is_stale(df)
    return row


def get_latest_rows_from_snapshot(universe: List[str]) -> pd.DataFrame:
    snap = load_indicator_snapshot()
    if snap.empty:
        return pd.DataFrame()
    uni = [sanitize_ticker(s) for s in universe]
    snap["Ticker"] = snap["Ticker"].astype(str).str.upper()
    return snap[snap["Ticker"].isin(uni)].copy()


def build_snapshot_from_cache(universe: List[str]) -> pd.DataFrame:
    rows = []
    for symbol in universe:
        df = load_cached_symbol(symbol)
        feat = compute_features(df)
        if feat.empty:
            continue
        rows.append(latest_row_with_symbol(symbol, feat))
    out = pd.DataFrame(rows)
    if not out.empty:
        save_indicator_snapshot(out)
    return out


def filter_scan(snapshot_df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    if snapshot_df.empty:
        return snapshot_df

    df = snapshot_df.copy()
    conds = [
        df["Close"] > params["min_price"],
        df["AVG_VOL20"] > params["min_avg_vol"],
        df["TSI_323"] > params["tsi323_min"],
        df["TSI_747"] > params["tsi747_min"],
        df["CCI15"] > params["cci_min"],
        df["BBPCT"] > params["bbpct_min"],
        df["EXT10"] > params["ext10_min"],
    ]
    if params.get("require_above_sma20", False):
        conds.append(df["Close"] > df["SMA20"])
    if params.get("require_above_sma50", False):
        conds.append(df["Close"] > df["SMA50"])
    if params.get("require_cci_roll", False):
        conds.append(df["CCI15_DELTA"] < 0)
    if params.get("require_optionable", False):
        conds.append(df["OptionableAssumed"] == True)

    mask = conds[0]
    for c in conds[1:]:
        mask &= c
    out = df[mask].copy()

    rank_by = params.get("rank_by", "DiamondScore")
    sort_col = "BBPCT" if rank_by == "%B" else "DIAMOND_SCORE"
    return out.sort_values([sort_col, "DIAMOND_SCORE", "BBPCT"], ascending=[False, False, False]).reset_index(drop=True)


def diamond_grade(score: float) -> str:
    if pd.isna(score):
        return "N/A"
    if score >= 85:
        return "A+ Fire"
    if score >= 78:
        return "A Strong"
    if score >= 70:
        return "B+ Watch"
    if score >= 62:
        return "B Developing"
    if score >= 52:
        return "C Warm"
    if score >= 40:
        return "D Weak"
    return "F Not a diamond"


def condition_report(last: pd.Series, params: Dict) -> pd.DataFrame:
    checks = [
        ("Optionable proxy", bool(last.get("OptionableAssumed", False)), params.get("require_optionable", False)),
        ("Close > 5", safe_float(last.get("Close")) > params["min_price"], True),
        ("AvgVol20 > 1M", safe_float(last.get("AVG_VOL20")) > params["min_avg_vol"], True),
        ("Close > SMA20", safe_float(last.get("Close")) > safe_float(last.get("SMA20")), params.get("require_above_sma20", False)),
        ("Close > SMA50", safe_float(last.get("Close")) > safe_float(last.get("SMA50")), params.get("require_above_sma50", False)),
        ("TSI(3,2,3)", safe_float(last.get("TSI_323")) > params["tsi323_min"], True),
        ("TSI(7,4,7)", safe_float(last.get("TSI_747")) > params["tsi747_min"], True),
        ("CCI(15)", safe_float(last.get("CCI15")) > params["cci_min"], True),
        ("CCI rollover", safe_float(last.get("CCI15_DELTA")) < 0, params.get("require_cci_roll", False)),
        ("%B(20,2)", safe_float(last.get("BBPCT")) > params["bbpct_min"], True),
        ("Close / SMA10 > 1.03", safe_float(last.get("EXT10")) > params["ext10_min"], True),
    ]
    rows = []
    for name, passed, active in checks:
        rows.append({"Condition": name, "Active": "Yes" if active else "No", "Pass": "✅" if passed else "❌"})
    return pd.DataFrame(rows)

# =============================================================================
# CHARTS
# =============================================================================
def empirical_bell_figure(series: pd.Series, current_value: float, prev_value: float, title: str) -> go.Figure:
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
        (0, 15, "#00c853"), (15, 30, "#90ee90"), (30, 45, "#fff59d"),
        (45, 55, "#ffffff"), (55, 70, "#ffe082"), (70, 85, "#ef9a9a"), (85, 100, "#ff1744")
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

    fig.update_layout(title=title, height=230, margin=dict(l=20, r=20, t=45, b=20), xaxis_title="Score (0-100)", yaxis_visible=False)
    fig.update_xaxes(range=[0, 100])
    return fig


def price_panel(df: pd.DataFrame, symbol: str) -> go.Figure:
    px = df["Adj Close"].fillna(df["Close"]) if "Adj Close" in df.columns else df["Close"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=px, mode="lines", name="Price"))
    for col, dash in [("SMA10", "dash"), ("SMA20", "dot"), ("SMA50", "dashdot")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(dash=dash)))
    fig.update_layout(title=f"{symbol} Price", height=320, margin=dict(l=20, r=20, t=45, b=20))
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("Universe")
    universe_mode = st.radio("Universe source", ["Built-in live universe", "Upload CSV/TXT", "Paste tickers"], index=0)
    preset_name = st.selectbox("Built-in universe preset", list(BUILTIN_UNIVERSES.keys()), index=0)
    uploaded = st.file_uploader("Optional ticker CSV/TXT", type=["csv", "txt"])
    universe_text = st.text_area("Or paste tickers", value="")

    st.header("Refresh")
    batch_size = st.slider("Yahoo batch size", 5, 20, 10, 1)
    sleep_s = st.slider("Sleep between batches (sec)", 0.0, 2.0, 0.7, 0.1)
    refresh_period = st.selectbox("Download window", ["1y", "2y", "3y"], index=1)
    max_refresh_symbols = st.slider("Max symbols to refresh now", 20, 250, 120, 10)

    st.header("Scan logic")
    scan_profile_name = st.selectbox("Preset scan logic", list(SCAN_PROFILES.keys()), index=0)
    fire_mode = st.checkbox("Fire mode", value=False, help="Adds the sure-fire layer on top.")
    profile_values = apply_scan_profile(scan_profile_name, fire_mode)

    st.caption("The default Watchlist preset matches your StockCharts logic.")
    min_price = st.number_input("Min price", value=float(profile_values["min_price"]), step=0.5)
    min_avg_vol = st.number_input("Min avg vol20", value=float(profile_values["min_avg_vol"]), step=100_000.0)
    require_optionable = st.checkbox("Require optionable proxy", value=bool(profile_values["require_optionable"]))
    require_above_sma20 = st.checkbox("Close > SMA20", value=bool(profile_values["require_above_sma20"]))
    require_above_sma50 = st.checkbox("Close > SMA50", value=bool(profile_values["require_above_sma50"]))
    tsi323_min = st.slider("TSI(3,2,3) >", 50.0, 100.0, float(profile_values["tsi323_min"]), 1.0)
    tsi747_min = st.slider("TSI(7,4,7) >", 40.0, 100.0, float(profile_values["tsi747_min"]), 1.0)
    cci_min = st.slider("CCI(15) >", 50.0, 200.0, float(profile_values["cci_min"]), 1.0)
    require_cci_roll = st.checkbox("Today's CCI < yesterday's CCI", value=bool(profile_values["require_cci_roll"]))
    bbpct_min = st.slider("%B(20,2) >", 0.50, 1.50, float(profile_values["bbpct_min"]), 0.01)
    ext10_min = st.slider("Close / SMA10 - 1 >", 0.00, 0.10, float(profile_values["ext10_min"]), 0.005)
    rank_by = st.selectbox("Rank by", ["DiamondScore", "%B"], index=0 if profile_values.get("rank_by") == "DiamondScore" else 1)
    top_n = st.slider("Top results", 5, 100, 25, 5)

# =============================================================================
# INPUT PARSING
# =============================================================================
def parse_universe() -> List[str]:
    symbols: List[str] = []
    if universe_mode == "Built-in live universe":
        symbols.extend(BUILTIN_UNIVERSES.get(preset_name, DEFAULT_UNIVERSE))
    elif universe_mode == "Upload CSV/TXT" and uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                u = pd.read_csv(uploaded)
                tick_col = next((c for c in u.columns if c.lower() == "ticker"), u.columns[0])
                symbols.extend(u[tick_col].astype(str).tolist())
            else:
                symbols.extend(uploaded.read().decode("utf-8", errors="ignore").replace("\n", ",").split(","))
        except Exception:
            pass
    elif universe_mode == "Paste tickers" and universe_text.strip():
        symbols.extend(universe_text.replace("\n", ",").split(","))

    cleaned = []
    for s in symbols:
        t = sanitize_ticker(s)
        if t and t not in cleaned:
            cleaned.append(t)
    return cleaned or BUILTIN_UNIVERSES.get(preset_name, DEFAULT_UNIVERSE)


universe = parse_universe()
scan_params = {
    "min_price": min_price,
    "min_avg_vol": min_avg_vol,
    "require_optionable": require_optionable,
    "require_above_sma20": require_above_sma20,
    "require_above_sma50": require_above_sma50,
    "tsi323_min": tsi323_min,
    "tsi747_min": tsi747_min,
    "cci_min": cci_min,
    "require_cci_roll": require_cci_roll,
    "bbpct_min": bbpct_min,
    "ext10_min": ext10_min,
    "rank_by": rank_by,
}

# =============================================================================
# ACTIONS
# =============================================================================
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("Refresh cache"):
        with st.spinner("Refreshing Yahoo into local cache..."):
            updated, failed, skipped = refresh_cache(
                universe,
                batch_size=batch_size,
                sleep_s=sleep_s,
                period=refresh_period,
                max_refresh_symbols=max_refresh_symbols,
            )
        st.success(f"Refresh complete. Updated: {updated}, failed: {failed}, skipped fresh: {skipped}")
with col2:
    if st.button("Scan now"):
        st.session_state["run_scan"] = True
with col3:
    st.caption(f"Universe size: {len(universe)} | Cached symbols: {len(list(DAILY_CACHE.glob('*.parquet')))}")
    st.caption(f"Default search preset: Watchlist | Active logic: {'Fire' if fire_mode else scan_profile_name}")

# =============================================================================
# SCAN
# =============================================================================
if st.session_state.get("run_scan", False):
    snap = get_latest_rows_from_snapshot(universe)
    if snap.empty or len(snap) < max(5, int(0.6 * len(universe))):
        snap = build_snapshot_from_cache(universe)

    results = filter_scan(snap, scan_params)

    if results.empty:
        st.warning("No results matched the current logic. Refresh cache or loosen thresholds.")
    else:
        st.subheader("Diamond scan results")
        display = results.head(top_n).copy()
        display = display[[
            "Ticker", "Group", "Date", "Close", "AVG_VOL20", "TSI_323", "TSI_747", "CCI15", "CCI15_DELTA",
            "BBPCT", "EXT10", "HEAT_SCORE", "DIAMOND_SCORE", "ULTIMATE_SCORE", "Fresh"
        ]].rename(columns={
            "AVG_VOL20": "AvgVol20",
            "CCI15_DELTA": "CCIΔ",
            "BBPCT": "%B",
            "EXT10": "Ext10",
            "HEAT_SCORE": "HeatScore",
            "DIAMOND_SCORE": "DiamondScore",
            "ULTIMATE_SCORE": "UltimateScore",
        })
        display["Grade"] = display["DiamondScore"].apply(diamond_grade)
        st.dataframe(
            display.style.format({
                "Close": "{:.2f}", "AvgVol20": "{:,.0f}", "TSI_323": "{:.1f}", "TSI_747": "{:.1f}",
                "CCI15": "{:.1f}", "CCIΔ": "{:.1f}", "%B": "{:.2f}", "Ext10": "{:.3f}",
                "HeatScore": "{:.1f}", "DiamondScore": "{:.1f}", "UltimateScore": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True,
        )
        snap_file = SNAPSHOT_CACHE / f"diamond_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results.to_csv(snap_file, index=False)
        st.download_button("Download scan snapshot", results.to_csv(index=False).encode("utf-8"), snap_file.name, "text/csv")

        selected = st.selectbox("Analyze ticker from results", display["Ticker"].tolist(), index=0)
        raw = load_cached_symbol(selected)
        feat = compute_features(raw)
        if not feat.empty:
            base = feat.dropna(how="all")
            last = base.iloc[-1]
            prev = base.iloc[-2] if len(base) > 1 else pd.Series(dtype=float)

            st.subheader(f"{selected} diamond analysis")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Diamond Score", f"{safe_float(last.get('DIAMOND_SCORE')):.1f}", f"{safe_float(last.get('DIAMOND_SCORE') - prev.get('DIAMOND_SCORE', np.nan)):.1f}" if not prev.empty else None)
            m2.metric("Grade", diamond_grade(safe_float(last.get("DIAMOND_SCORE"))))
            m3.metric("Ultimate Score", f"{safe_float(last.get('ULTIMATE_SCORE')):.1f}")
            m4.metric("TSI(3,2,3)", f"{safe_float(last.get('TSI_323')):.1f}")
            m5.metric("CCI(15)", f"{safe_float(last.get('CCI15')):.1f}")

            st.plotly_chart(price_panel(feat.tail(252), selected), use_container_width=True)
            b1, b2 = st.columns(2)
            with b1:
                st.plotly_chart(empirical_bell_figure(feat["ULTIMATE_SCORE"], safe_float(last.get("ULTIMATE_SCORE")), safe_float(prev.get("ULTIMATE_SCORE", np.nan)), f"{selected} Ultimate Bell"), use_container_width=True)
                st.plotly_chart(empirical_bell_figure(feat["TSI_323_SCORE"], safe_float(last.get("TSI_323_SCORE")), safe_float(prev.get("TSI_323_SCORE", np.nan)), f"{selected} TSI(3,2,3) Bell"), use_container_width=True)
            with b2:
                st.plotly_chart(empirical_bell_figure(feat["CCI15_SCORE"], safe_float(last.get("CCI15_SCORE")), safe_float(prev.get("CCI15_SCORE", np.nan)), f"{selected} CCI(15) Bell"), use_container_width=True)
                st.plotly_chart(empirical_bell_figure(feat["BBPCT_SCORE"], safe_float(last.get("BBPCT_SCORE")), safe_float(prev.get("BBPCT_SCORE", np.nan)), f"{selected} %B Bell"), use_container_width=True)

# =============================================================================
# SINGLE STOCK EVALUATION
# =============================================================================
st.markdown("---")
st.subheader("Single stock evaluation")
col_a, col_b, col_c = st.columns([2, 1, 3])
with col_a:
    eval_text = st.text_input("Paste a ticker to grade on the diamond scale", value="SMH")
with col_b:
    eval_refresh = st.button("Refresh ticker")
with col_c:
    st.caption("Example: paste SMH and it will compute a Diamond Score grade, show which watchlist conditions pass, and render the bells.")

eval_symbol = sanitize_ticker(eval_text)
if eval_symbol:
    if eval_refresh or load_cached_symbol(eval_symbol).empty:
        ok, msg = refresh_one_symbol(eval_symbol, period=refresh_period)
        if ok:
            st.success(f"{eval_symbol} refreshed into cache.")
        else:
            st.warning(f"{eval_symbol} refresh issue: {msg}. Using cached data if available.")

    eval_raw = load_cached_symbol(eval_symbol)
    eval_feat = compute_features(eval_raw)
    if eval_feat.empty:
        st.info("No cached history yet for that ticker. Click Refresh ticker.")
    else:
        base = eval_feat.dropna(how="all")
        last = latest_row_with_symbol(eval_symbol, eval_feat)
        prev = base.iloc[-2] if len(base) > 1 else pd.Series(dtype=float)

        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Diamond Score", f"{safe_float(last.get('DIAMOND_SCORE')):.1f}")
        g2.metric("Grade", diamond_grade(safe_float(last.get("DIAMOND_SCORE"))))
        g3.metric("Heat Score", f"{safe_float(last.get('HEAT_SCORE')):.1f}")
        g4.metric("Ultimate Score", f"{safe_float(last.get('ULTIMATE_SCORE')):.1f}")
        g5.metric("Watchlist hit", "Yes" if bool(last.get("MEETS_WATCHLIST", False)) else "No")

        st.dataframe(
            condition_report(last, scan_params),
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(price_panel(eval_feat.tail(252), eval_symbol), use_container_width=True)
        e1, e2 = st.columns(2)
        with e1:
            st.plotly_chart(empirical_bell_figure(eval_feat["DIAMOND_SCORE"], safe_float(last.get("DIAMOND_SCORE")), safe_float(prev.get("DIAMOND_SCORE", np.nan)), f"{eval_symbol} Diamond Bell"), use_container_width=True)
            st.plotly_chart(empirical_bell_figure(eval_feat["TSI_323_SCORE"], safe_float(last.get("TSI_323_SCORE")), safe_float(prev.get("TSI_323_SCORE", np.nan)), f"{eval_symbol} TSI(3,2,3) Bell"), use_container_width=True)
        with e2:
            st.plotly_chart(empirical_bell_figure(eval_feat["CCI15_SCORE"], safe_float(last.get("CCI15_SCORE")), safe_float(prev.get("CCI15_SCORE", np.nan)), f"{eval_symbol} CCI(15) Bell"), use_container_width=True)
            st.plotly_chart(empirical_bell_figure(eval_feat["BBPCT_SCORE"], safe_float(last.get("BBPCT_SCORE")), safe_float(prev.get("BBPCT_SCORE", np.nan)), f"{eval_symbol} %B Bell"), use_container_width=True)

# =============================================================================
# FOOTER STATUS
# =============================================================================
status_df = load_refresh_status()
if not status_df.empty:
    with st.expander("Recent refresh log"):
        st.dataframe(status_df.sort_values("ts", ascending=False).head(50), use_container_width=True, hide_index=True)
else:
    st.info("Default mode is now the Watchlist search. Refresh cache first, then scan locally. The scan itself uses cached indicator rows instead of hitting Yahoo.")
