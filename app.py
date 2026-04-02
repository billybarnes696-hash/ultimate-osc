
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

def _clean_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def _find_price_column(df: pd.DataFrame):
    cols = [str(c).lower().strip() for c in df.columns]
    original = dict(zip(cols, df.columns))
    for cand in ["close", "adj close", "adj_close", "rsp", "value", "last"]:
        if cand in original:
            return original[cand]
    for c in df.columns:
        if str(c).lower() not in {"date", "datetime", "timestamp", "name", "symbol"}:
            return c
    return None

def load_rsp_history(symbol, years, zip_file):
    if zip_file is not None:
        try:
            zip_file.seek(0)
        except Exception:
            pass

        with zipfile.ZipFile(zip_file) as z:
            rsp_candidates = [n for n in z.namelist() if "rsp" in n.lower() and n.lower().endswith(".csv")]
            for name in rsp_candidates:
                try:
                    raw = pd.read_csv(z.open(name))
                    raw.columns = [str(c).strip() for c in raw.columns]

                    date_col = None
                    for cand in ["Date", "date", "DATE", "Datetime", "datetime", "timestamp"]:
                        if cand in raw.columns:
                            date_col = cand
                            break
                    if date_col is None:
                        continue

                    price_col = _find_price_column(raw)
                    if price_col is None:
                        continue

                    df = pd.DataFrame(
                        {"price": _clean_numeric_series(raw[price_col])},
                        index=pd.to_datetime(raw[date_col], errors="coerce"),
                    )
                    df = df[~df.index.isna()].sort_index()
                    df = df.dropna(subset=["price"])
                    df = df[~df.index.duplicated(keep="last")]
                    if len(df) > 50:
                        return df
                except Exception:
                    continue

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
            out = out.dropna(subset=["price"])
            if len(out) > 50:
                return out
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
    return (x - lo) / denom * 100

def build_oscillator(df):
    px = pd.to_numeric(df["price"], errors="coerce").dropna().astype(float)
    r = rsi(px)
    s = stochastic(px)
    b = bb_percent(px)
    t = tsi(px)

    comp = pd.concat([r, s, b, t], axis=1)
    comp.columns = ["RSI", "Stoch", "BB%", "TSI"]
    comp = comp.replace([np.inf, -np.inf], np.nan)

    score = comp.mean(axis=1, skipna=True)
    score = normalize(score)
    score = score.rolling(5).mean().clip(0, 100)

    return pd.DataFrame({"price": px, "osc": score}).dropna()

def backtest(df, long_th, short_th, hold=5):
    px = df["price"].astype(float)
    osc = df["osc"].astype(float)
    px_ret = px.pct_change().fillna(0.0)

    position = 0
    bars_left = 0
    rets = []

    for i in range(len(df)):
        if position == 0:
            if osc.iloc[i] >= long_th:
                position = 1
                bars_left = hold
            elif osc.iloc[i] <= short_th:
                position = -1
                bars_left = hold

        rets.append(px_ret.iloc[i] * position)

        if position != 0:
            bars_left -= 1
            if bars_left <= 0:
                position = 0

    return (1 + pd.Series(rets, index=df.index)).cumprod()

price_df = load_rsp_history(symbol, years, zip_file)
if price_df is None:
    st.error("Could not load history for RSP. Upload breadth zip containing rsp.csv.")
    st.stop()

data = build_oscillator(price_df)
if data.empty or len(data) < 250:
    st.error("Loaded price history, but not enough valid numeric data remained after cleaning.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Oscillator", "Backtest Sweet Spots", "Monte Carlo", "Data Check"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["osc"], name="Ultimate Oscillator"))
    fig.add_hline(y=20)
    fig.add_hline(y=80)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Rows used in oscillator: {len(data):,}")

with tab2:
    st.subheader("Threshold Sweet Spot Search")
    long_candidates = [20, 22, 24, 26, 28, 30]
    short_candidates = [70, 72, 74, 76, 78, 80]
    hold_candidates = [3, 5, 8]

    results = []
    for long_th in long_candidates:
        for short_th in short_candidates:
            for hold in hold_candidates:
                eq = backtest(data, long_th, short_th, hold=hold)
                total = float(eq.iloc[-1]) if len(eq) else np.nan
                results.append((long_th, short_th, hold, total))

    res = pd.DataFrame(results, columns=["Long Trigger", "Short Trigger", "Hold Bars", "Return"])
    res["Return"] = pd.to_numeric(res["Return"], errors="coerce")
    res = res.dropna(subset=["Return"]).sort_values("Return", ascending=False).reset_index(drop=True)
    st.dataframe(res, width="stretch")

    if len(res):
        top = res.iloc[0]
        st.write(f"Top combo: long {int(top['Long Trigger'])}, short {int(top['Short Trigger'])}, hold {int(top['Hold Bars'])}, terminal equity {top['Return']:.3f}")

with tab3:
    st.subheader("Monte Carlo Stress Test")
    eq = backtest(data, 20, 80, hold=5)
    rets = eq.pct_change().dropna()

    sims = []
    for _ in range(200):
        r = rets.sample(frac=1, replace=True).reset_index(drop=True)
        sims.append((1 + r).cumprod())

    sims = pd.concat(sims, axis=1)
    fig = go.Figure()
    for col in sims.columns[:30]:
        fig.add_trace(go.Scatter(y=sims[col], mode="lines", opacity=0.25, showlegend=False))
    st.plotly_chart(fig, width="stretch")

with tab4:
    st.subheader("Data Check")
    st.write("Loaded price head:")
    st.dataframe(price_df.head(10), width="stretch")
    st.write("Dtypes:")
    st.write(price_df.dtypes.astype(str))
    st.write("Cleaned oscillator frame head:")
    st.dataframe(data.head(10), width="stretch")
