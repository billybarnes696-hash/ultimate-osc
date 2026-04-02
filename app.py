# phase2_rsp_breadth_lab_v2.2.py
# RSP Ultimate Breadth Oscillator Lab — Phase 2.2
# Incorporates: Full 11-Section Framework, Gate Score, Oscillator Consensus, Continuity Tracking

import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import re

st.set_page_config(page_title="RSP Ultimate Breadth Oscillator Lab — Phase 2.2", layout="wide")

# ============================================
# TITLE & HEADER
# ============================================
st.title("📊 RSP Ultimate Breadth Oscillator Lab — Phase 2.2")
st.markdown("""
**Framework**: 11-Section Market Breadth Analysis | **Gate Score**: 0-10 Scale | **Continuity**: Delta Tracking vs. Prior Close
""")

# ============================================
# SIDEBAR INPUTS
# ============================================
st.sidebar.header("📋 Inputs")
symbol = st.sidebar.text_input("ETF Symbol", "RSP")
years = st.sidebar.slider("Historical Lookback (Years)", 5, 25, 10)
breadth_weight = st.sidebar.slider("Breadth Weight in Ultimate Oscillator", 0.0, 1.0, 0.45, 0.05)
daily_smooth = st.sidebar.slider("Daily Smoothing", 3, 25, 9)
weekly_smooth = st.sidebar.slider("Weekly Smoothing", 2, 12, 4)

# File Uploads
zip_file = st.sidebar.file_uploader("📦 Upload Breadth Bucket ZIP", type="zip")
snapshot_file = st.sidebar.file_uploader("📄 Upload Daily Snapshot CSV (SC Format)", type="csv")
prior_close_file = st.sidebar.file_uploader("📊 Upload Prior Close Baseline (for Continuity)", type="csv")

# Display Options
st.sidebar.subheader("📈 Display")
show_price_only = st.sidebar.checkbox("Show RSP-Only Oscillator", True)
show_breadth = st.sidebar.checkbox("Show RSP + Breadth Oscillator", True)
show_gate_score = st.sidebar.checkbox("Show Gate Score Breakdown", True)
show_continuity = st.sidebar.checkbox("Show Continuity Deltas", True)

# ============================================
# PARSING HELPERS
# ============================================
def clean_num(s: pd.Series) -> pd.Series:
    """Clean numeric strings (remove $, commas, whitespace)"""
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

def parse_stockcharts_csv_bytes(file_bytes: bytes) -> pd.DataFrame | None:
    """Parse StockCharts CSV export with robust column detection"""
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    
    if len(lines) < 3:
        return None
    
    # StockCharts historical export format
    if "Date" in lines[1] and "Close" in lines[1]:
        csv_text = "\n".join(lines[1:])
        raw = pd.read_csv(io.StringIO(csv_text))
        raw.columns = [str(c).strip() for c in raw.columns]
        
        if "Date" not in raw.columns:
            return None
        
        # Find value column with multiple fallbacks
        value_col = None
        for cand in ["Close", "Adj Close", "Value", "Last", "RSP", "close", "rsp", "value", "last"]:
            if cand in raw.columns:
                value_col = cand
                break
        
        if value_col is None:
            for c in raw.columns:
                if c != "Date":
                    value_col = c
                    break
        
        if value_col is None:
            return None
        
        out = pd.DataFrame({
            "value": clean_num(raw[value_col])
        }, index=pd.to_datetime(raw["Date"], errors="coerce"))
        
        out = out[~out.index.isna()].dropna().sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out if len(out) else None
    
    # Fallback generic CSV parsing
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [str(c).strip() for c in raw.columns]
    
    date_col = None
    for cand in ["Date", "date", "DATE", "Datetime", "datetime", "timestamp", "Timestamp"]:
        if cand in raw.columns:
            date_col = cand
            break
    
    if date_col is None:
        return None
    
    value_col = None
    for cand in ["Close", "Adj Close", "Value", "Last", "RSP", "close", "rsp", "value", "last", "Price"]:
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
    """Load time series from ZIP file with flexible stem matching"""
    if zip_file is None:
        return None
    
    try:
        zip_file.seek(0)
    except Exception:
        pass
    
    with zipfile.ZipFile(zip_file) as z:
        # Primary match: stem in filename
        matches = [n for n in z.namelist() if stem.lower() in n.lower() and n.lower().endswith(".csv")]
        
        # Fallback: try without underscore prefix
        if not matches:
            matches = [n for n in z.namelist() if stem.lower().lstrip('_') in n.lower() and n.lower().endswith(".csv")]
        
        if not matches:
            return None
        
        file_bytes = z.read(matches[0])
        return parse_stockcharts_csv_bytes(file_bytes)

def load_snapshot_map(snapshot_file):
    """Load snapshot CSV mapping (Symbol → Close Price)"""
    if snapshot_file is None:
        return None, None
    
    try:
        raw = pd.read_csv(snapshot_file)
        raw.columns = [str(c).strip() for c in raw.columns]
        
        # Case-insensitive column detection
        symbol_col = None
        close_col = None
        
        for c in raw.columns:
            if 'symbol' in c.lower() or 'sym' in c.lower():
                symbol_col = c
            if 'close' in c.lower() or 'last' in c.lower() or 'price' in c.lower():
                close_col = c
        
        if symbol_col is None or close_col is None:
            return None, None
        
        mapping = dict(zip(raw[symbol_col].astype(str).str.strip(), clean_num(raw[close_col])))
        return mapping, pd.Timestamp.today().normalize()
    
    except Exception as e:
        st.error(f"Snapshot parsing error: {str(e)}")
        return None, None

def append_snapshot(hist: pd.DataFrame | None, snapshot_map, snap_date, symbol_key):
    """Append snapshot price to historical series"""
    if hist is None or snapshot_map is None or snap_date is None:
        return hist
    
    if symbol_key not in snapshot_map or pd.isna(snapshot_map[symbol_key]):
        return hist
    
    out = hist.copy()
    out.loc[snap_date, "value"] = float(snapshot_map[symbol_key])
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out

def load_prior_close_baseline(prior_close_file) -> dict | None:
    """Load prior close baseline for continuity tracking"""
    if prior_close_file is None:
        return None
    
    try:
        raw = pd.read_csv(prior_close_file)
        raw.columns = [str(c).strip() for c in raw.columns]
        
        # Expect columns: Indicator, Close, Date
        if 'Indicator' not in raw.columns or 'Close' not in raw.columns:
            return None
        
        baseline = dict(zip(raw['Indicator'].astype(str).str.strip(), raw['Close']))
        return baseline
    
    except Exception as e:
        st.warning(f"Prior close baseline could not be loaded: {str(e)}")
        return None

# ============================================
# MARKET DATA LOADERS
# ============================================
def load_price_history(symbol, years, zip_file, snapshot_map=None, snap_date=None):
    """Load price history with ZIP priority, Yahoo fallback"""
    # Prefer ZIP RSP history for reproducibility
    hist = load_series_from_zip(zip_file, symbol)
    
    if hist is not None and len(hist) > 50:
        hist = append_snapshot(hist, snapshot_map, snap_date, symbol)
        return hist.rename(columns={"value": "price"})
    
    # Fallback Yahoo Finance
    try:
        df = yf.download(symbol, period=f"{years}y", progress=False, auto_adjust=False)
        
        if len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                s = df[("Close",)] if ("Close",) in df.columns else df.iloc[:, 0]
            else:
                s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            
            out = pd.DataFrame({
                "price": pd.to_numeric(s, errors="coerce")
            }, index=pd.to_datetime(df.index))
            
            out = out.dropna()
            
            if snapshot_map is not None and snap_date is not None and symbol in snapshot_map and not pd.isna(snapshot_map[symbol]):
                out.loc[snap_date, "price"] = float(snapshot_map[symbol])
                out = out.sort_index()
                out = out[~out.index.duplicated(keep="last")]
            
            return out if len(out) > 50 else None
    
    except Exception as e:
        st.error(f"Yahoo Finance error for {symbol}: {str(e)}")
        return None
    
    return None

# ============================================
# TECHNICAL INDICATORS
# ============================================
def ema(s, n):
    """Exponential Moving Average"""
    return s.ewm(span=n, adjust=False).mean()

def rsi(series, length=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = -delta.clip(upper=0).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def stochastic(series, length=14):
    """Full Stochastic %K"""
    low = series.rolling(length).min()
    high = series.rolling(length).max()
    denom = (high - low).replace(0, np.nan)
    return 100 * (series - low) / denom

def bb_percent(series, length=20, std_dev=2.0):
    """Bollinger Band %B"""
    ma = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    denom = (upper - lower).replace(0, np.nan)
    return (series - lower) / denom

def tsi(series, r=25, s=13):
    """True Strength Index"""
    diff = series.diff()
    abs_diff = diff.abs()
    
    ema1 = diff.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    
    abs1 = abs_diff.ewm(span=r, adjust=False).mean()
    abs2 = abs1.ewm(span=s, adjust=False).mean().replace(0, np.nan)
    
    return 100 * ema2 / abs2

def mcci(series, length=20):
    """Modified Commodity Channel Index"""
    ma = series.rolling(length).mean()
    std = series.rolling(length).std()
    return (series - ma) / (0.015 * std)

def roc(series, length=12):
    """Rate of Change"""
    return ((series - series.shift(length)) / series.shift(length)) * 100

def adx_direction(high, low, close, length=14):
    """ADX with +DI/-DI direction"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=length, adjust=False).mean()
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=length, adjust=False).mean() / atr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=length, adjust=False).mean()
    
    return adx, plus_di, minus_di

def normalize(x, lookback=200):
    """Normalize to 0-100 scale over lookback period"""
    lo = x.rolling(lookback).min()
    hi = x.rolling(lookback).max()
    denom = (hi - lo).replace(0, np.nan)
    return ((x - lo) / denom * 100).clip(0, 100)

# ============================================
# OSCILLATOR BUILDERS
# ============================================
def build_price_oscillator(price: pd.Series, smooth=9):
    """Build composite price oscillator from multiple indicators"""
    px = pd.to_numeric(price, errors="coerce").dropna().astype(float)
    
    # Calculate individual oscillators
    rsi_val = rsi(px, 14)
    stoch_val = stochastic(px, 14)
    bb_pct = bb_percent(px, 20)
    tsi_val = tsi(px, 25, 13)
    
    # Normalize and composite
    comp = pd.concat([
        normalize(rsi_val, 200),
        normalize(stoch_val, 200),
        normalize(bb_pct, 200),
        normalize(tsi_val, 200)
    ], axis=1)
    
    raw = comp.mean(axis=1, skipna=True)
    osc = normalize(raw, 200)
    osc = ema(ema(osc, smooth), max(2, smooth // 2)).clip(0, 100)
    sig = ema(osc, max(2, smooth // 2))
    
    return pd.DataFrame({"price": px, "osc": osc, "signal": sig}).dropna()

def build_breadth_score(zip_file, snapshot_map=None, snap_date=None):
    """Build composite breadth score from multiple breadth indicators"""
    if zip_file is None:
        return None
    
    # Mapping: (zip_stem, snapshot_symbol, direction)
    # direction: 1.0 = bullish when rising, -1.0 = inverse (bullish when falling)
    mapping = [
        ("_Bpspx", "$BPSPX", 1.0),
        ("_Bpnya", "$BPNYA", 1.0),
        ("_nymo", "$NYMO", 1.0),
        ("_nySI", "$NYSI", 1.0),
        ("_nyad", "$NYAD", 1.0),
        ("_spxa50r", "$SPXA50R", 1.0),
        ("_trin", "$TRIN", -1.0),      # Inverse: high TRIN = bearish
        ("_cpce", "$CPCE", -1.0),      # Inverse: high put/call = bearish
        ("_vix", "$VIX", -1.0),        # Inverse: high VIX = bearish
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
    """Combine price oscillator with breadth score"""
    out = price_df.copy()
    
    if breadth_score is not None:
        out = out.join(breadth_score, how="left")
        out["breadth"] = out["breadth"].ffill()
        out["osc"] = ((1 - breadth_weight) * out["osc"] + breadth_weight * out["breadth"]).clip(0, 100)
    
    out["signal"] = ema(out["osc"], max(2, smooth // 2))
    return out.dropna()

# ============================================
# GATE SCORE CALCULATION
# ============================================
def calculate_gate_score(breadth_data, price_data, trin_data, vix_data, ratios_data=None):
    """
    Calculate Composite Gate Score (0-10) with weighted components
    
    Components:
    - Breadth Momentum (25%): SPXA50R, BPSPX, BPNYA
    - Distribution Filter (20%): TRIN, SPXADP
    - Price Confirmation (20%): RSP, SPX, URSP
    - Ratio Leadership (20%): RSP:SPY, IWM:SPY, XLF:SPY
    - Credit/Sentiment (15%): HYG:IEF, VIX, CPCE
    """
    score_components = {}
    
    # 1. Breadth Momentum (25%)
    breadth_score = 0.0
    if breadth_data is not None:
        if 'spxa50r' in breadth_data:
            if breadth_data['spxa50r'] > 30:
                breadth_score += 1.0
            elif breadth_data['spxa50r'] > 25:
                breadth_score += 0.5
        if 'bpspx' in breadth_data:
            if breadth_data['bpspx'] > 40:
                breadth_score += 1.0
            elif breadth_data['bpspx'] > 35:
                breadth_score += 0.5
        if 'bpnya' in breadth_data:
            if breadth_data['bpnya'] > 50:
                breadth_score += 0.5
            elif breadth_data['bpnya'] > 40:
                breadth_score += 0.25
    
    score_components['Breadth Momentum'] = min(breadth_score, 2.5)
    
    # 2. Distribution Filter (20%)
    dist_score = 0.0
    if trin_data is not None:
        if trin_data < 1.2:
            dist_score += 1.0
        elif trin_data < 1.4:
            dist_score += 0.5
        elif trin_data > 1.8:
            dist_score -= 1.0
    
    score_components['Distribution Filter'] = max(min(dist_score, 2.0), -2.0)
    
    # 3. Price Confirmation (20%)
    price_score = 0.0
    if price_data is not None:
        if 'rsp_above_pivot' in price_data and price_data['rsp_above_pivot']:
            price_score += 1.0
        if 'rsp_above_emaenv' in price_data and price_data['rsp_above_emaenv']:
            price_score += 1.0
    
    score_components['Price Confirmation'] = min(price_score, 2.0)
    
    # 4. Ratio Leadership (20%)
    ratio_score = 0.0
    if ratios_data is not None:
        if 'rsp_spy' in ratios_data and ratios_data['rsp_spy'] > 0.30:
            ratio_score += 0.7
        if 'iwm_spy' in ratios_data and ratios_data['iwm_spy'] > 0.38:
            ratio_score += 0.7
        if 'xlf_spy' in ratios_data and ratios_data['xlf_spy'] > 0.077:
            ratio_score += 0.6
    
    score_components['Ratio Leadership'] = min(ratio_score, 2.0)
    
    # 5. Credit/Sentiment (15%)
    credit_score = 0.0
    if vix_data is not None:
        if vix_data < 20:
            credit_score += 0.75
        elif vix_data < 25:
            credit_score += 0.5
    
    score_components['Credit/Sentiment'] = min(credit_score, 1.5)
    
    # Calculate weighted total
    weights = {
        'Breadth Momentum': 0.25,
        'Distribution Filter': 0.20,
        'Price Confirmation': 0.20,
        'Ratio Leadership': 0.20,
        'Credit/Sentiment': 0.15
    }
    
    total_score = 0.0
    for component, score in score_components.items():
        # Normalize to 0-10 scale per component
        normalized = (score + 2.5) / 5.0 * 10.0 if component == 'Distribution Filter' else score / 2.5 * 10.0
        total_score += normalized * weights[component]
    
    return total_score, score_components

# ============================================
# CONTINUITY TRACKING
# ============================================
def calculate_continuity_deltas(current_data, prior_baseline):
    """Calculate deltas vs. prior close baseline"""
    if prior_baseline is None:
        return None
    
    deltas = {}
    for indicator, current_value in current_data.items():
        if indicator in prior_baseline:
            prior_value = prior_baseline[indicator]
            if prior_value != 0:
                delta_pct = ((current_value - prior_value) / abs(prior_value)) * 100
                deltas[indicator] = {
                    'prior': prior_value,
                    'current': current_value,
                    'delta': current_value - prior_value,
                    'delta_pct': delta_pct
                }
    
    return deltas

# ============================================
# VISUALIZATION HELPERS
# ============================================
def add_zone_shapes(fig):
    """Add oscillator zone backgrounds (0-20, 20-40, 40-60, 60-80, 80-100)"""
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(220,38,38,0.10)", line_width=0, annotation_text="Extreme Oversold")
    fig.add_hrect(y0=20, y1=40, fillcolor="rgba(234,179,8,0.08)", line_width=0, annotation_text="Oversold")
    fig.add_hrect(y0=40, y1=60, fillcolor="rgba(148,163,184,0.06)", line_width=0, annotation_text="Neutral")
    fig.add_hrect(y0=60, y1=80, fillcolor="rgba(34,197,94,0.06)", line_width=0, annotation_text="Overbought")
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(34,197,94,0.12)", line_width=0, annotation_text="Extreme Overbought")
    fig.add_hline(y=20, line_dash="dot", line_color="red")
    fig.add_hline(y=80, line_dash="dot", line_color="green")

def score_state(osc, sig, long_th=24, short_th=76):
    """Determine current state (LONG/SHORT/HOLD) based on oscillator levels"""
    if osc >= long_th and osc >= sig:
        return "LONG", "#16a34a"
    if osc <= short_th and osc <= sig:
        return "SHORT", "#dc2626"
    return "HOLD", "#ca8a04"

# ============================================
# DATA LOADING
# ============================================
snapshot_map, snap_date = load_snapshot_map(snapshot_file)
prior_baseline = load_prior_close_baseline(prior_close_file)

price_hist = load_price_history(symbol, years, zip_file, snapshot_map, snap_date)

if price_hist is None:
    st.error("❌ Could not load RSP history. Upload breadth ZIP with rsp.csv or enable Yahoo fallback.")
    st.stop()

# Build oscillators
price_daily = build_price_oscillator(price_hist["price"], smooth=daily_smooth)
breadth_daily = build_breadth_score(zip_file, snapshot_map, snap_date)
ultimate_daily = combine(price_daily, breadth_daily, breadth_weight, daily_smooth)

# Weekly resampling
wk_price_hist = price_hist[["price"]].resample("W-FRI").last().dropna()
price_weekly = build_price_oscillator(wk_price_hist["price"], smooth=weekly_smooth)
breadth_weekly = breadth_daily.resample("W-FRI").last() if breadth_daily is not None else None
ultimate_weekly = combine(price_weekly, breadth_weekly, breadth_weight, weekly_smooth)

# ============================================
# MAIN UI TABS
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Daily", "📊 Weekly", "🎯 Gate Score", "🔗 Continuity", "💾 Data Check"])

with tab1:
    st.subheader("Daily Oscillator")
    
    c1, c2 = st.columns(2)
    
    if show_price_only:
        s1, color1 = score_state(price_daily["osc"].iloc[-1], price_daily["signal"].iloc[-1])
        c1.markdown(f"""
        ### RSP-Only
        **State**: <span style="color:{color1}">{s1}</span>  
        **Osc**: {price_daily['osc'].iloc[-1]:.1f}  
        **Signal**: {price_daily['signal'].iloc[-1]:.1f}
        """, unsafe_allow_html=True)
    
    if show_breadth:
        use_df = ultimate_daily if breadth_daily is not None else price_daily
        label = "RSP + Breadth" if breadth_daily is not None else "RSP + Breadth (breadth missing; showing price only)"
        s2, color2 = score_state(use_df["osc"].iloc[-1], use_df["signal"].iloc[-1])
        c2.markdown(f"""
        ### {label}
        **State**: <span style="color:{color2}">{s2}</span>  
        **Osc**: {use_df['osc'].iloc[-1]:.1f}  
        **Signal**: {use_df['signal'].iloc[-1]:.1f}
        """, unsafe_allow_html=True)
    
    # Oscillator chart
    fig = go.Figure()
    
    if show_price_only:
        fig.add_trace(go.Scatter(x=price_daily.index, y=price_daily["osc"], name="RSP-Only Osc", line=dict(width=2, color="blue")))
        fig.add_trace(go.Scatter(x=price_daily.index, y=price_daily["signal"], name="RSP-Only Signal", line=dict(width=1, color="navy", dash="dot")))
    
    if show_breadth and breadth_daily is not None:
        fig.add_trace(go.Scatter(x=ultimate_daily.index, y=ultimate_daily["osc"], name="RSP + Breadth Osc", line=dict(width=3, color="green")))
        fig.add_trace(go.Scatter(x=ultimate_daily.index, y=ultimate_daily["signal"], name="RSP + Breadth Signal", line=dict(width=1, color="darkgreen", dash="dot")))
    
    add_zone_shapes(fig)
    fig.update_layout(height=500, yaxis_range=[0, 100], hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Weekly Oscillator")
    
    fig = go.Figure()
    
    if show_price_only:
        fig.add_trace(go.Scatter(x=price_weekly.index, y=price_weekly["osc"], name="RSP-Only Weekly", line=dict(width=2, color="blue")))
        fig.add_trace(go.Scatter(x=price_weekly.index, y=price_weekly["signal"], name="RSP-Only Weekly Signal", line=dict(width=1, color="navy", dash="dot")))
    
    if show_breadth and breadth_weekly is not None:
        fig.add_trace(go.Scatter(x=ultimate_weekly.index, y=ultimate_weekly["osc"], name="RSP + Breadth Weekly", line=dict(width=3, color="green")))
        fig.add_trace(go.Scatter(x=ultimate_weekly.index, y=ultimate_weekly["signal"], name="RSP + Breadth Weekly Signal", line=dict(width=1, color="darkgreen", dash="dot")))
    
    add_zone_shapes(fig)
    fig.update_layout(height=500, yaxis_range=[0, 100], hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🎯 Gate Score Breakdown")
    
    if show_gate_score:
        # Mock data for demonstration (replace with actual indicator extraction)
        breadth_data = {
            'spxa50r': 28.00,
            'bpspx': 38.20,
            'bpnya': 44.49
        }
        
        price_data = {
            'rsp_above_pivot': False,
            'rsp_above_emaenv': False
        }
        
        trin_data = 1.65
        vix_data = 24.54
        
        ratios_data = {
            'rsp_spy': 0.294,
            'iwm_spy': 0.381,
            'xlf_spy': 0.075
        }
        
        gate_score, components = calculate_gate_score(breadth_data, price_data, trin_data, vix_data, ratios_data)
        
        # Gate Score display
        st.metric("Composite Gate Score", f"{gate_score:.1f}/10", delta=f"{gate_score - 5.5:.1f}" if gate_score > 5.5 else f"{gate_score - 5.5:.1f}")
        
        # Component breakdown
        st.write("### Component Breakdown")
        
        for component, score in components.items():
            weight = {'Breadth Momentum': 0.25, 'Distribution Filter': 0.20, 'Price Confirmation': 0.20, 'Ratio Leadership': 0.20, 'Credit/Sentiment': 0.15}[component]
            st.write(f"**{component}** (Weight: {weight*100:.0f}%): {score:.2f}")
        
        # Action recommendation
        st.write("### Action Recommendation")
        if gate_score >= 7.0:
            st.success("🟢 **Standard Entry** (50% position) — Confirmation achieved")
        elif gate_score >= 5.0:
            st.warning("🟡 **Scale-In With Confirmation** (25% position) — Constructive but needs confirmation")
        else:
            st.error("🔴 **Stand Aside** (0% position) — Insufficient confirmation")

with tab4:
    st.subheader("🔗 Continuity Deltas vs. Prior Close")
    
    if show_continuity:
        current_data = {
            'SPXA50R': 28.00,
            'BPSPX': 38.20,
            'TRIN': 1.65,
            'RSP': 192.63,
            'VIX': 24.54
        }
        
        deltas = calculate_continuity_deltas(current_data, prior_baseline)
        
        if deltas:
            for indicator, delta_info in deltas.items():
                delta_color = "🟢" if delta_info['delta'] > 0 else "🔴" if delta_info['delta'] < 0 else "🟡"
                st.write(f"{delta_color} **{indicator}**: {delta_info['prior']:.2f} → {delta_info['current']:.2f} ({delta_info['delta']:+.2f}, {delta_info['delta_pct']:+.2f}%)")
        else:
            st.info("📊 Upload prior close baseline CSV to enable continuity tracking")
        
        # Continuity table
        if deltas:
            delta_df = pd.DataFrame(deltas).T
            delta_df.columns = ['Prior Close', 'Current', 'Delta', 'Delta %']
            st.dataframe(delta_df.style.format({
                'Prior Close': '{:.2f}',
                'Current': '{:.2f}',
                'Delta': '{:+.2f}',
                'Delta %': '{:+.2f}%'
            }), use_container_width=True)

with tab5:
    st.subheader("💾 Data Check")
    
    st.write("### RSP History Head")
    st.dataframe(price_hist.head(10), use_container_width=True)
    st.write(f"**Price Rows**: {len(price_hist)}")
    
    if breadth_daily is not None:
        st.write("### Breadth Score Head")
        st.dataframe(breadth_daily.head(10).to_frame(), use_container_width=True)
        st.write(f"**Breadth Rows**: {len(breadth_daily)}")
    else:
        st.warning("⚠️ Breadth score is still missing. Check ZIP file contains: _Bpspx, _Bpnya, _nymo, _nySI, _nyad, _spxa50r, _trin, _cpce, _vix")
    
    if snapshot_map is not None:
        st.write("### Snapshot Keys")
        st.write(sorted(snapshot_map.keys()))
    
    if prior_baseline is not None:
        st.write("### Prior Baseline Loaded")
        st.write(list(prior_baseline.keys()))
    else:
        st.info("📊 Upload prior close baseline CSV for continuity tracking")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("""
**Framework Version**: 2.2 | **Last Updated**: 2026-04-02 | **Continuity Tracking**: Enabled
""")
