
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="RSP Ultimate Breadth Oscillator Lab", layout="wide")
st.title("RSP Ultimate Breadth Oscillator Lab")

st.sidebar.header("Inputs")
symbol = st.sidebar.text_input("ETF symbol", "RSP")
years = st.sidebar.slider("Historical lookback (years)", 5, 25, 25)
breadth_weight = st.sidebar.slider("Breadth weight in ultimate oscillator", 0.0, 1.0, 0.45)
zip_file = st.sidebar.file_uploader("Upload breadth bucket zip", type="zip")

daily_smooth = st.sidebar.slider("Daily oscillator smoothing", 3, 25, 9)
weekly_smooth = st.sidebar.slider("Weekly oscillator smoothing", 2, 12, 4)

def _clean_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def _find_date_col(df: pd.DataFrame):
    for cand in ["Date", "date", "DATE", "Datetime", "datetime", "timestamp"]:
        if cand in df.columns:
            return cand
    return None

def _find_price_col(df: pd.DataFrame):
    cols = [str(c).lower().strip() for c in df.columns]
    original = dict(zip(cols, df.columns))
    for cand in ["close", "adj close", "adj_close", "rsp", "value", "last"]:
        if cand in original:
            return original[cand]
    for c in df.columns:
        if str(c).lower() not in {"date", "datetime", "timestamp", "name", "symbol"}:
            return c
    return None

def _find_file_in_zip(z, stem):
    matches = [n for n in z.namelist() if stem.lower() in n.lower() and n.lower().endswith(".csv")]
    return matches[0] if matches else None

def load_series_from_zip(zip_file, stem, value_name):
    try:
        zip_file.seek(0)
    except Exception:
        pass
    with zipfile.ZipFile(zip_file) as z:
        name = _find_file_in_zip(z, stem)
        if not name:
            return None
        raw = pd.read_csv(z.open(name))
        raw.columns = [str(c).strip() for c in raw.columns]
        date_col = _find_date_col(raw)
        if date_col is None:
            return None
        val_col = _find_price_col(raw)
        if val_col is None:
            return None
        out = pd.DataFrame(
            {value_name: _clean_numeric_series(raw[val_col])},
            index=pd.to_datetime(raw[date_col], errors="coerce"),
        )
        out = out[~out.index.isna()].sort_index()
        out = out.dropna()
        out = out[~out.index.duplicated(keep="last")]
        return out if len(out) > 50 else None

def load_rsp_history(symbol, years, zip_file):
    if zip_file is not None:
        df = load_series_from_zip(zip_file, "rsp", "price")
        if df is not None:
            return df
    try:
        df = yf.download(symbol, period=f"{years}y", progress=False, auto_adjust=False)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                if ("Close", symbol) in df.columns:
                    s = df[("Close", symbol)]
                elif ("Adj Close", symbol) in df.columns:
                    s = df[("Adj Close", symbol)]
                else:
                    s = df.iloc[:, 0]
            else:
                if "Close" in df.columns:
                    s = df["Close"]
                elif "Adj Close" in df.columns:
                    s = df["Adj Close"]
                else:
                    s = df.iloc[:, 0]
            out = pd.DataFrame({"price": pd.to_numeric(s, errors="coerce")}, index=pd.to_datetime(df.index))
            out = out.dropna()
            return out if len(out) > 50 else None
    except Exception:
        pass
    return None

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = -delta.clip(upper=0).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def stochastic(series, length=14):
    low = series.rolling(length).min()
    high = series.rolling(length).max()
    denom = (high - low).replace(0, np.nan)
    return 100 * (series - low) / denom

def bb_percent(series, length=20):
    ma = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    denom = (upper - lower).replace(0, np.nan)
    return (series - lower) / denom * 100

def tsi(series, r=25, s=13):
    diff = series.diff()
    abs_diff = diff.abs()
    ema1 = diff.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    abs1 = abs_diff.ewm(span=r, adjust=False).mean()
    abs2 = abs1.ewm(span=s, adjust=False).mean().replace(0, np.nan)
    return 100 * ema2 / abs2

def normalize(x, lookback=200):
    lo = x.rolling(lookback).min()
    hi = x.rolling(lookback).max()
    denom = (hi - lo).replace(0, np.nan)
    return ((x - lo) / denom * 100).clip(0, 100)

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def build_price_oscillator(px, smooth=9):
    px = pd.to_numeric(px, errors="coerce").dropna().astype(float)
    comp = pd.concat(
        [
            rsi(px, 14),
            stochastic(px, 14),
            bb_percent(px, 20),
            tsi(px, 25, 13),
        ],
        axis=1,
    )
    comp.columns = ["RSI", "Stoch", "BB%", "TSI"]
    raw = comp.mean(axis=1, skipna=True)
    osc = normalize(raw, lookback=200)
    osc = ema(ema(osc, smooth), max(2, smooth // 2))
    osc = osc.clip(0, 100)
    sig = ema(osc, max(2, smooth // 2))
    return pd.DataFrame({"price": px, "osc": osc, "signal": sig}).dropna()

def build_breadth_score(zip_file):
    if zip_file is None:
        return None
    pieces = []
    mapping = [
        ("_Bpspx", 1),
        ("_Bpnya", 1),
        ("_nymo", 1),
        ("_nySI", 1),
        ("_nyad", 1),
        ("_spxa50r", 1),
        ("_trin", -1),
        ("_cpce", -1),
        ("vix", -1),
    ]
    for stem, direction in mapping:
        s = load_series_from_zip(zip_file, stem, stem)
        if s is None:
            continue
        col = s.columns[0]
        vals = s[col].astype(float) * direction
        vals = normalize(vals, lookback=200)
        pieces.append(vals.rename(stem))
    if not pieces:
        return None
    comp = pd.concat(pieces, axis=1).dropna(how="all")
    breadth = comp.mean(axis=1, skipna=True)
    breadth = ema(ema(breadth, 7), 4).clip(0, 100)
    return breadth.rename("breadth")

def combine_price_and_breadth(price_df, breadth_score, breadth_weight=0.45, smooth=9):
    out = price_df.copy()
    if breadth_score is not None:
        out = out.join(breadth_score, how="left")
        out["breadth"] = out["breadth"].ffill()
        pw = 1.0 - breadth_weight
        out["osc"] = (pw * out["osc"] + breadth_weight * out["breadth"]).clip(0, 100)
        out["signal"] = ema(out["osc"], max(2, smooth // 2))
    return out.dropna()

def build_weekly_from_daily(daily_df, smooth=4):
    wk = daily_df[["price"]].resample("W-FRI").last().dropna()
    wk = build_price_oscillator(wk["price"], smooth=smooth)
    return wk

def backtest_crosses(df, long_th, short_th, hold=5):
    osc = df["osc"].astype(float)
    px = df["price"].astype(float)
    ret = px.pct_change().fillna(0.0)
    pos = 0
    bars_left = 0
    out = []
    for i in range(1, len(df)):
        prev = osc.iloc[i-1]
        cur = osc.iloc[i]
        if pos == 0:
            if prev < long_th and cur >= long_th:
                pos = 1
                bars_left = hold
            elif prev > short_th and cur <= short_th:
                pos = -1
                bars_left = hold
        out.append(ret.iloc[i] * pos)
        if pos != 0:
            bars_left -= 1
            if bars_left <= 0:
                pos = 0
    return (1 + pd.Series(out, index=df.index[1:])).cumprod()

price_df = load_rsp_history(symbol, years, zip_file)
if price_df is None:
    st.error("Could not load history for RSP. Upload breadth zip containing rsp.csv.")
    st.stop()

daily_price = build_price_oscillator(price_df["price"], smooth=daily_smooth)
breadth_score = build_breadth_score(zip_file)
daily = combine_price_and_breadth(daily_price, breadth_score, breadth_weight, smooth=daily_smooth)

weekly = build_weekly_from_daily(price_df, smooth=weekly_smooth)
if breadth_score is not None:
    weekly_breadth = breadth_score.resample("W-FRI").last()
    weekly = combine_price_and_breadth(weekly, weekly_breadth, breadth_weight, smooth=weekly_smooth)

if daily.empty:
    st.error("Not enough daily data after cleaning.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Daily Oscillator", "Weekly Oscillator", "Backtest Sweet Spots", "Data Check"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily["osc"], name="Daily Ultimate Oscillator", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=daily.index, y=daily["signal"], name="Daily Signal", line=dict(width=1)))
    fig.add_hline(y=20)
    fig.add_hline(y=80)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Daily rows used: {len(daily):,}")

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["osc"], name="Weekly Ultimate Oscillator", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["signal"], name="Weekly Signal", line=dict(width=1)))
    fig.add_hline(y=20)
    fig.add_hline(y=80)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Weekly rows used: {len(weekly):,}")

with tab3:
    long_candidates = [20, 22, 24, 26, 28, 30]
    short_candidates = [70, 72, 74, 76, 78, 80]
    hold_candidates = [3, 5, 8]
    mode = st.selectbox("Backtest timeframe", ["Daily", "Weekly"])
    bt_df = daily if mode == "Daily" else weekly

    results = []
    for long_th in long_candidates:
        for short_th in short_candidates:
            for hold in hold_candidates:
                eq = backtest_crosses(bt_df, long_th, short_th, hold)
                total = float(eq.iloc[-1]) if len(eq) else np.nan
                results.append((mode, long_th, short_th, hold, total))
    res = pd.DataFrame(results, columns=["Timeframe", "Long Trigger", "Short Trigger", "Hold Bars", "Return"])
    res["Return"] = pd.to_numeric(res["Return"], errors="coerce")
    res = res.dropna(subset=["Return"]).sort_values("Return", ascending=False).reset_index(drop=True)
    st.dataframe(res, width="stretch")

with tab4:
    st.write("Daily price head")
    st.dataframe(price_df.head(10), width="stretch")
    if breadth_score is not None:
        st.write("Breadth score head")
        st.dataframe(breadth_score.head(10).to_frame(), width="stretch")
