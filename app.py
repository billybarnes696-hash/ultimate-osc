
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="RSP Ultimate Breadth Oscillator Lab", layout="wide")

st.title("RSP Ultimate Breadth Oscillator Lab")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Inputs")

symbol = st.sidebar.text_input("ETF symbol", "RSP")
years = st.sidebar.slider("Historical lookback (years)", 5, 25, 25)

breadth_weight = st.sidebar.slider("Breadth weight in ultimate oscillator", 0.0, 1.0, 0.45)

zip_file = st.sidebar.file_uploader("Upload breadth bucket zip", type="zip")

# ---------------------------
# Load RSP History
# ---------------------------
def load_rsp_history(symbol, years, zip_file):

    # FIRST: try to load RSP from uploaded ZIP
    if zip_file is not None:
        with zipfile.ZipFile(zip_file) as z:
            for name in z.namelist():
                if "rsp" in name.lower() and name.endswith(".csv"):

                    df = pd.read_csv(z.open(name))
                    df.columns = [c.lower() for c in df.columns]

                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df.set_index("date", inplace=True)

                    if "close" in df.columns:
                        df = df[["close"]].rename(columns={"close": "price"})
                    elif "rsp" in df.columns:
                        df = df[["rsp"]].rename(columns={"rsp": "price"})
                    else:
                        continue

                    return df

    # SECOND: try Yahoo Finance
    try:
        df = yf.download(symbol, period=f"{years}y", progress=False)
        if len(df) > 0:
            df = df[["Close"]].rename(columns={"Close": "price"})
            return df
    except:
        pass

    return None

# ---------------------------
# Indicators
# ---------------------------
def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = -delta.clip(upper=0).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def stochastic(series, length=14):
    low = series.rolling(length).min()
    high = series.rolling(length).max()
    return 100 * (series - low) / (high - low)

def bb_percent(series, length=20):
    ma = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (series - lower) / (upper - lower) * 100

def tsi(series, r=25, s=13):
    diff = series.diff()
    abs_diff = abs(diff)
    ema1 = diff.ewm(span=r).mean()
    ema2 = ema1.ewm(span=s).mean()
    abs1 = abs_diff.ewm(span=r).mean()
    abs2 = abs1.ewm(span=s).mean()
    return 100 * ema2 / abs2

def normalize(x):
    return (x - x.rolling(200).min()) / (x.rolling(200).max() - x.rolling(200).min()) * 100

# ---------------------------
# Build Oscillator
# ---------------------------
def build_oscillator(df):

    r = rsi(df.price)
    s = stochastic(df.price)
    b = bb_percent(df.price)
    t = tsi(df.price)

    comp = pd.concat([r, s, b, t], axis=1)
    comp.columns = ["RSI","Stoch","BB%","TSI"]

    score = comp.mean(axis=1)
    score = normalize(score)
    score = score.rolling(5).mean()

    return score

# ---------------------------
# Backtest
# ---------------------------
def backtest(df, osc, long_th, short_th, hold=5):

    position = 0
    bars = 0
    returns = []

    for i in range(1,len(df)):

        if position == 0:
            if osc.iloc[i] > long_th:
                position = 1
                bars = hold
            elif osc.iloc[i] < short_th:
                position = -1
                bars = hold

        if position != 0:
            bars -= 1

        if bars <= 0:
            position = 0

        ret = df.price.pct_change().iloc[i] * position
        returns.append(ret)

    equity = (1 + pd.Series(returns)).cumprod()
    return equity

# ---------------------------
# Load Data
# ---------------------------
df = load_rsp_history(symbol, years, zip_file)

if df is None:
    st.error("Could not load history for RSP. Upload breadth zip containing rsp.csv.")
    st.stop()

osc = build_oscillator(df)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Oscillator", "Backtest Sweet Spots", "Monte Carlo"])

# ---------------------------
# Oscillator Chart
# ---------------------------
with tab1:

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=osc, name="Ultimate Oscillator"))
    fig.add_hline(y=20)
    fig.add_hline(y=80)

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Backtest Grid
# ---------------------------
with tab2:

    st.subheader("Threshold Sweet Spot Search")

    long_candidates = [20,22,24,26,28,30]
    short_candidates = [70,72,74,76,78,80]

    results = []

    for long_th in long_candidates:
        for short_th in short_candidates:
            eq = backtest(df, osc, long_th, short_th)
            total = eq.iloc[-1]
            results.append((long_th, short_th, total))

    res = pd.DataFrame(results, columns=["Long Trigger","Short Trigger","Return"])
    res = res.sort_values("Return", ascending=False)

    st.dataframe(res)

# ---------------------------
# Monte Carlo
# ---------------------------
with tab3:

    st.subheader("Monte Carlo Stress Test")

    eq = backtest(df, osc, 20, 80)
    rets = eq.pct_change().dropna()

    sims = []

    for i in range(200):
        r = rets.sample(frac=1, replace=True).reset_index(drop=True)
        sims.append((1+r).cumprod())

    sims = pd.concat(sims, axis=1)

    fig = go.Figure()

    for col in sims.columns[:30]:
        fig.add_trace(go.Scatter(y=sims[col], mode="lines", opacity=0.3, showlegend=False))

    st.plotly_chart(fig, use_container_width=True)
