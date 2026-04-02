
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="RSP Ultimate Breadth Oscillator Lab — Phase 2.1", layout="wide")
st.title("RSP Ultimate Breadth Oscillator Lab — Phase 2.1")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Inputs")
symbol = st.sidebar.text_input("ETF symbol", "RSP")
years = st.sidebar.slider("Historical lookback (years)", 5, 25, 10)
breadth_weight = st.sidebar.slider("Breadth weight in ultimate oscillator", 0.0, 1.0, 0.45, 0.05)
daily_smooth = st.sidebar.slider("Daily smoothing", 3, 25, 9)
weekly_smooth = st.sidebar.slider("Weekly smoothing", 2, 12, 4)
zip_file = st.sidebar.file_uploader("Upload breadth bucket zip", type="zip")
snapshot_file = st.sidebar.file_uploader("Upload daily snapshot csv (SC format)", type="csv")

st.sidebar.subheader("Display")
show_price_only = st.sidebar.checkbox("Show RSP-only oscillator", True)
show_breadth = st.sidebar.checkbox("Show RSP + breadth oscillator", True)

# ---------------------------
# Parsing helpers
# ---------------------------
def clean_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("$", "", regex=False)
         .str.strip(),
        errors="coerce",
    )

def parse_stockcharts_csv_bytes(file_bytes: bytes) -> pd.DataFrame | None:
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None

    # StockCharts historical export:
    # line 1: "$BPSPX, Daily"
    # line 2: "Date, Open, High, Low, Close, Volume"
    # remaining lines: data rows
    if "Date" in lines[1] and "Close" in lines[1]:
        csv_text = "\n".join(lines[1:])
        raw = pd.read_csv(io.StringIO(csv_text))
        raw.columns = [str(c).strip() for c in raw.columns]

        if "Date" not in raw.columns:
            return None

        value_col = None
        for cand in ["Close", "Adj Close", "Value", "Last", "RSP"]:
            if cand in raw.columns:
                value_col = cand
                break
        if value_col is None:
            # use first non-date column as fallback
            for c in raw.columns:
                if c != "Date":
                    value_col = c
                    break

        out = pd.DataFrame({
            "value": clean_num(raw[value_col])
        }, index=pd.to_datetime(raw["Date"], errors="coerce"))
        out = out[~out.index.isna()].dropna().sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out if len(out) else None

    # fallback generic csv
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = None
    for cand in ["Date", "date", "DATE", "Datetime", "datetime", "timestamp"]:
        if cand in raw.columns:
            date_col = cand
            break
    if date_col is None:
        return None

    value_col = None
    for cand in ["Close", "Adj Close", "Value", "Last", "RSP", "close", "rsp", "value", "last"]:
        if cand in raw.columns:
            value_col = cand
            break
    if value_col is None:
        for c in raw.columns:
            if str(c).lower() not in {"date", "datetime", "timestamp", "name", "symbol"}:
                value_col = c
                break
    if value_col is None:
        return None

    out = pd.DataFrame({
        "value": clean_num(raw[value_col])
    }, index=pd.to_datetime(raw[date_col], errors="coerce"))
    out = out[~out.index.isna()].dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out if len(out) else None

def load_series_from_zip(zip_file, stem: str) -> pd.DataFrame | None:
    if zip_file is None:
        return None
    try:
        zip_file.seek(0)
    except Exception:
        pass
    with zipfile.ZipFile(zip_file) as z:
        matches = [n for n in z.namelist() if stem.lower() in n.lower() and n.lower().endswith(".csv")]
        if not matches:
            return None
        file_bytes = z.read(matches[0])
        return parse_stockcharts_csv_bytes(file_bytes)

def load_snapshot_map(snapshot_file):
    if snapshot_file is None:
        return None, None
    raw = pd.read_csv(snapshot_file)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "Symbol" not in raw.columns or "Close" not in raw.columns:
        return None, None
    mapping = dict(zip(raw["Symbol"].astype(str).str.strip(), clean_num(raw["Close"])))
    return mapping, pd.Timestamp.today().normalize()

def append_snapshot(hist: pd.DataFrame | None, snapshot_map, snap_date, symbol_key):
    if hist is None or snapshot_map is None or snap_date is None:
        return hist
    if symbol_key not in snapshot_map or pd.isna(snapshot_map[symbol_key]):
        return hist
    out = hist.copy()
    out.loc[snap_date, "value"] = float(snapshot_map[symbol_key])
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out

# ---------------------------
# Market data loaders
# ---------------------------
def load_price_history(symbol, years, zip_file, snapshot_map=None, snap_date=None):
    # Prefer zip RSP history for reproducibility
    hist = load_series_from_zip(zip_file, symbol)
    if hist is not None and len(hist) > 50:
        hist = append_snapshot(hist, snapshot_map, snap_date, symbol)
        return hist.rename(columns={"value": "price"})

    # Fallback Yahoo
    try:
        df = yf.download(symbol, period=f"{years}y", progress=False, auto_adjust=False)
        if len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                s = df[("Close", symbol)] if ("Close", symbol) in df.columns else df.iloc[:, 0]
            else:
                s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            out = pd.DataFrame({"price": pd.to_numeric(s, errors="coerce")}, index=pd.to_datetime(df.index))
            out = out.dropna()
            if snapshot_map is not None and snap_date is not None and symbol in snapshot_map and not pd.isna(snapshot_map[symbol]):
                out.loc[snap_date, "price"] = float(snapshot_map[symbol])
                out = out.sort_index()
                out = out[~out.index.duplicated(keep="last")]
            return out if len(out) > 50 else None
    except Exception:
        return None
    return None

# ---------------------------
# Indicators
# ---------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

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

# ---------------------------
# Oscillators
# ---------------------------
def build_price_oscillator(price: pd.Series, smooth=9):
    px = pd.to_numeric(price, errors="coerce").dropna().astype(float)
    comp = pd.concat([
        rsi(px, 14),
        stochastic(px, 14),
        bb_percent(px, 20),
        tsi(px, 25, 13)
    ], axis=1)
    raw = comp.mean(axis=1, skipna=True)
    osc = normalize(raw, 200)
    osc = ema(ema(osc, smooth), max(2, smooth // 2)).clip(0, 100)
    sig = ema(osc, max(2, smooth // 2))
    return pd.DataFrame({"price": px, "osc": osc, "signal": sig}).dropna()

def build_breadth_score(zip_file, snapshot_map=None, snap_date=None):
    if zip_file is None:
        return None
    # stem in zip, symbol in snapshot, direction
    mapping = [
        ("_Bpspx", "$BPSPX", 1.0),
        ("_Bpnya", "$BPNYA", 1.0),
        ("_nymo", "$NYMO", 1.0),
        ("_nySI", "$NYSI", 1.0),
        ("_nyad", "$NYAD", 1.0),
        ("_spxa50r", "$SPXA50R", 1.0),
        ("_trin", "$TRIN", -1.0),
        ("_cpce", "$CPCE", -1.0),
        ("_vix", "$VIX", -1.0),
    ]
    pieces = []
    for stem, snap_symbol, direction in mapping:
        hist = load_series_from_zip(zip_file, stem)
        if hist is None:
            continue
        hist = append_snapshot(hist, snapshot_map, snap_date, snap_symbol)
        vals = hist["value"].astype(float) * direction
        vals = normalize(vals, 200)
        pieces.append(vals.rename(stem))

    if not pieces:
        return None

    comp = pd.concat(pieces, axis=1).dropna(how="all")
    breadth = comp.mean(axis=1, skipna=True)
    breadth = ema(ema(breadth, 7), 4).clip(0, 100)
    return breadth.rename("breadth")

def combine(price_df, breadth_score=None, breadth_weight=0.45, smooth=9):
    out = price_df.copy()
    if breadth_score is not None:
        out = out.join(breadth_score, how="left")
        out["breadth"] = out["breadth"].ffill()
        out["osc"] = ((1 - breadth_weight) * out["osc"] + breadth_weight * out["breadth"]).clip(0, 100)
        out["signal"] = ema(out["osc"], max(2, smooth // 2))
    return out.dropna()

# ---------------------------
# Backtest + display helpers
# ---------------------------
def resample_weekly(price_hist):
    return price_hist[["price"]].resample("W-FRI").last().dropna()

def backtest_crosses(df, long_th, short_th, hold=5, mode="long_cash"):
    osc = df["osc"].astype(float)
    px = df["price"].astype(float)
    ret = px.pct_change().fillna(0.0)
    pos = 0
    bars = 0
    out = []
    for i in range(1, len(df)):
        prev = osc.iloc[i - 1]
        cur = osc.iloc[i]
        if pos == 0:
            if prev < long_th and cur >= long_th:
                pos = 1
                bars = hold
            elif mode == "long_short" and prev > short_th and cur <= short_th:
                pos = -1
                bars = hold

        out.append(ret.iloc[i] * pos)

        if pos != 0:
            bars -= 1
            if bars <= 0:
                pos = 0
    return (1 + pd.Series(out, index=df.index[1:])).cumprod()

def score_state(osc, sig, long_th=24, short_th=76):
    if osc >= long_th and osc >= sig:
        return "LONG", "#16a34a"
    if osc <= short_th and osc <= sig:
        return "SHORT", "#dc2626"
    return "HOLD", "#ca8a04"

def add_zone_shapes(fig):
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(220,38,38,0.10)", line_width=0)
    fig.add_hrect(y0=20, y1=40, fillcolor="rgba(234,179,8,0.08)", line_width=0)
    fig.add_hrect(y0=40, y1=60, fillcolor="rgba(148,163,184,0.06)", line_width=0)
    fig.add_hrect(y0=60, y1=80, fillcolor="rgba(34,197,94,0.06)", line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(34,197,94,0.12)", line_width=0)
    fig.add_hline(y=20)
    fig.add_hline(y=80)

# ---------------------------
# Load everything
# ---------------------------
snapshot_map, snap_date = load_snapshot_map(snapshot_file)
price_hist = load_price_history(symbol, years, zip_file, snapshot_map, snap_date)
if price_hist is None:
    st.error("Could not load RSP history. Upload breadth zip with rsp.csv.")
    st.stop()

price_daily = build_price_oscillator(price_hist["price"], smooth=daily_smooth)
breadth_daily = build_breadth_score(zip_file, snapshot_map, snap_date)
ultimate_daily = combine(price_daily, breadth_daily, breadth_weight, daily_smooth)

wk_price_hist = resample_weekly(price_hist)
price_weekly = build_price_oscillator(wk_price_hist["price"], smooth=weekly_smooth)
breadth_weekly = breadth_daily.resample("W-FRI").last() if breadth_daily is not None else None
ultimate_weekly = combine(price_weekly, breadth_weekly, breadth_weight, weekly_smooth)

# ---------------------------
# UI
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Daily", "Weekly", "Sweet Spots", "Data Check"])

with tab1:
    st.subheader("Daily Oscillator")
    c1, c2 = st.columns(2)
    if show_price_only:
        s1, color1 = score_state(price_daily["osc"].iloc[-1], price_daily["signal"].iloc[-1])
        c1.markdown(f"<div style='padding:14px;border-radius:10px;background:{color1};color:white;font-weight:700;text-align:center;'>RSP-only: {s1} | Osc {price_daily['osc'].iloc[-1]:.1f}</div>", unsafe_allow_html=True)
    if show_breadth:
        use_df = ultimate_daily if breadth_daily is not None else price_daily
        label = "RSP + Breadth" if breadth_daily is not None else "RSP + Breadth (breadth missing; showing price only)"
        s2, color2 = score_state(use_df["osc"].iloc[-1], use_df["signal"].iloc[-1])
        c2.markdown(f"<div style='padding:14px;border-radius:10px;background:{color2};color:white;font-weight:700;text-align:center;'>{label}: {s2} | Osc {use_df['osc'].iloc[-1]:.1f}</div>", unsafe_allow_html=True)

    fig = go.Figure()
    if show_price_only:
        fig.add_trace(go.Scatter(x=price_daily.index, y=price_daily["osc"], name="RSP-only Osc", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=price_daily.index, y=price_daily["signal"], name="RSP-only Signal", line=dict(width=1)))
    if show_breadth and breadth_daily is not None:
        fig.add_trace(go.Scatter(x=ultimate_daily.index, y=ultimate_daily["osc"], name="RSP + Breadth Osc", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=ultimate_daily.index, y=ultimate_daily["signal"], name="RSP + Breadth Signal", line=dict(width=1)))
    add_zone_shapes(fig)
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader("Weekly Oscillator")
    fig = go.Figure()
    if show_price_only:
        fig.add_trace(go.Scatter(x=price_weekly.index, y=price_weekly["osc"], name="RSP-only Weekly", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=price_weekly.index, y=price_weekly["signal"], name="RSP-only Weekly Signal", line=dict(width=1)))
    if show_breadth and breadth_weekly is not None:
        fig.add_trace(go.Scatter(x=ultimate_weekly.index, y=ultimate_weekly["osc"], name="RSP + Breadth Weekly", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=ultimate_weekly.index, y=ultimate_weekly["signal"], name="RSP + Breadth Weekly Signal", line=dict(width=1)))
    add_zone_shapes(fig)
    st.plotly_chart(fig, width="stretch")

with tab3:
    st.subheader("Sweet Spot Backtest")
    timeframe = st.selectbox("Backtest timeframe", ["Daily", "Weekly"])
    trade_mode = st.selectbox("Backtest mode", ["long_cash", "long_short"])
    bt = ultimate_daily if (timeframe == "Daily" and breadth_daily is not None) else (price_daily if timeframe == "Daily" else (ultimate_weekly if breadth_weekly is not None else price_weekly))
    rows = []
    for l in [20, 22, 24, 26, 28, 30]:
        for s in [70, 72, 74, 76, 78, 80]:
            for h in [3, 5, 8]:
                eq = backtest_crosses(bt, l, s, hold=h, mode=trade_mode)
                rows.append((l, s, h, float(eq.iloc[-1]) if len(eq) else np.nan))
    res = pd.DataFrame(rows, columns=["Long Trigger", "Short Trigger", "Hold Bars", "Return"])
    res["Return"] = pd.to_numeric(res["Return"], errors="coerce")
    res = res.dropna().sort_values("Return", ascending=False).reset_index(drop=True)
    st.dataframe(res, width="stretch")

with tab4:
    st.subheader("Data Check")
    st.write("RSP history head")
    st.dataframe(price_hist.head(10), width="stretch")
    st.write("Price rows:", len(price_hist))
    if breadth_daily is not None:
        st.write("Breadth score head")
        st.dataframe(breadth_daily.head(10).to_frame(), width="stretch")
        st.write("Breadth rows:", len(breadth_daily))
    else:
        st.warning("Breadth score is still missing.")
    if snapshot_map is not None:
        st.write("Snapshot keys")
        st.write(sorted(snapshot_map.keys()))
