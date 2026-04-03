import io
import zipfile
import traceback
import warnings
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="GLM Ultimate Oscillator Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

@dataclass
class AppConfig:
    lookback: int = 126
    smooth_price: float = 3.5
    smooth_breadth: float = 3.0
    breadth_weight: float = 0.70
    price_weight: float = 0.30
    target_vol: float = 0.12
    stop_loss_pct: float = 0.06
    min_quality_for_long: float = 0.45
    max_hold_bars: int = 20
    cost_bps: float = 3.0  # Added for UI configuration
    breadth_weights: Dict[str, float] = field(default_factory=lambda: {
        "SPXA50R": 2.1,
        "BPSPX": 1.7,
        "BPNYA": 1.2,
        "NYMO": 1.3,
        "NYSI": 1.0,
        "NYAD": 0.9,
        "TRIN": 0.7,
        "CPCE": 0.7,
        "OEXA50R": 0.7,
        "SPXADP": 0.8,
    })

def normalize_yf_close(df: pd.DataFrame, symbol: str) -> pd.Series:
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        for field_name in ("Close", "Adj Close"):
            if (field_name, symbol) in df.columns:
                s = df[(field_name, symbol)]
                break
        else:
            candidates = [c for c in df.columns if c[0] in ("Close", "Adj Close")]
            if not candidates:
                raise ValueError(f"Could not find Close/Adj Close for {symbol}")
            s = df[candidates[0]]
    else:
        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            raise ValueError(f"Could not find Close/Adj Close for {symbol}")
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = symbol.upper()
    return s

def parse_stockcharts_history_text(text: str) -> pd.Series:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return pd.Series(dtype=float)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return pd.Series(dtype=float)
    
    # Auto-detect delimiter
    first_line = lines[0]
    delimiter = "," if "," in first_line else "|"
    
    rows = []
    # Start from lines[1:] to skip header. Parsing checks will skip non-date lines.
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(delimiter)]
        if len(parts) < 5:
            continue
        try:
            dt = pd.to_datetime(parts[0], errors="coerce")
            close_val = pd.to_numeric(parts[4], errors="coerce")
            if pd.notna(dt) and pd.notna(close_val):
                rows.append((dt, float(close_val)))
        except Exception:
            continue
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({dt: val for dt, val in rows}).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.astype(float)

def parse_snapshot_csv(file_bytes: bytes) -> Dict[str, float]:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return {}
    colmap = {c.lower().strip(): c for c in df.columns}
    if "symbol" not in colmap or "close" not in colmap:
        return {}
    sym_col = colmap["symbol"]
    close_col = colmap["close"]
    out = {}
    tmp = df[[sym_col, close_col]].copy()
    tmp[sym_col] = tmp[sym_col].astype(str).str.strip().str.upper()
    tmp[close_col] = pd.to_numeric(tmp[close_col], errors="coerce")
    tmp = tmp.dropna()
    for _, row in tmp.iterrows():
        out[row[sym_col]] = float(row[close_col])
    return out

def snapshot_value(snapshot_map: Optional[Dict[str, float]], *aliases: str) -> Optional[float]:
    if not snapshot_map:
        return None
    for alias in aliases:
        key = alias.upper()
        if key in snapshot_map:
            return snapshot_map[key]
    return None

@st.cache_data(show_spinner=False)
def load_zip_bundle(zip_bytes: bytes) -> Dict[str, pd.Series]:
    bundle: Dict[str, pd.Series] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            stem = name.split("/")[-1].replace(".csv", "")
            text = zf.read(name).decode("utf-8", errors="ignore")
            s = parse_stockcharts_history_text(text)
            if not s.empty:
                bundle[stem] = s
    return bundle

@st.cache_data(show_spinner=False)
def fetch_price(symbol: str, period: str) -> pd.Series:
    # Added retry logic for robustness
    for attempt in range(3):
        try:
            df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
            return normalize_yf_close(df, symbol)
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            else:
                st.warning(f"Failed to load price data for {symbol} after 3 attempts.")
                return pd.Series(dtype=float)

class BellCurveTransform:
    @staticmethod
    def calculate(series: pd.Series, lookback: int, sigma: float, z_clip: float = 3.5) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").copy()
        s = s.interpolate(limit_direction="both").ffill().bfill()
        min_periods = max(20, int(lookback * 0.67))
        rolling_mean = s.rolling(lookback, min_periods=min_periods).mean()
        rolling_std = s.rolling(lookback, min_periods=min_periods).std().clip(lower=1e-8)
        z = ((s - rolling_mean) / rolling_std).clip(-z_clip, z_clip)
        cdf = norm.cdf(z) * 100.0
        if sigma > 0:
            smoothed = gaussian_filter1d(cdf.to_numpy(dtype=float), sigma=sigma)
            out = pd.Series(smoothed, index=s.index)
        else:
            out = pd.Series(cdf, index=s.index)
        out[rolling_mean.isna()] = np.nan
        return out

class RegimeDetector:
    def detect(self, price: pd.Series) -> pd.Series:
        px = pd.to_numeric(price, errors="coerce")
        rets = px.pct_change()
        trend_252 = px / px.shift(252) - 1.0
        trend_63 = px / px.shift(63) - 1.0
        vol_short = rets.rolling(20, min_periods=10).std() * np.sqrt(252)
        vol_long = rets.rolling(126, min_periods=60).std() * np.sqrt(252)
        vol_ratio = vol_short / vol_long.clip(lower=1e-8)
        dd = px / px.cummax() - 1.0
        regime = pd.Series("sideways", index=px.index, dtype=object)
        bull = (trend_252 > 0.03) & (trend_63 > -0.02) & (vol_ratio < 1.20) & (dd > -0.10)
        bear = (trend_252 < -0.03) | (trend_63 < -0.06) | (vol_ratio > 1.35) | (dd < -0.15)
        regime.loc[bull] = "bull"
        regime.loc[bear] = "bear"
        return regime.ffill().fillna("sideways")
    
    def thresholds(self, regime: str) -> Dict[str, float]:
        if regime == "bull":
            return {"long_enter": 22.0, "long_exit": 48.0, "risk_off": 92.0}
        if regime == "bear":
            return {"long_enter": 12.0, "long_exit": 35.0, "risk_off": 72.0}
        return {"long_enter": 18.0, "long_exit": 45.0, "risk_off": 82.0}

class BreadthThrustDetector:
    @staticmethod
    def calculate(breadth_curve: pd.Series, spxa50r_raw: Optional[pd.Series], bpspx_raw: Optional[pd.Series]) -> pd.Series:
        bo = pd.to_numeric(breadth_curve, errors="coerce")
        thrust = pd.Series(0.0, index=bo.index)
        thrust += (bo.diff(5) > 12).astype(float) * 0.40
        thrust += (bo.diff(10) > 18).astype(float) * 0.35
        thrust += ((bo > 35) & (bo.shift(3) < 20)).astype(float) * 0.25
        if spxa50r_raw is not None:
            s50 = pd.to_numeric(spxa50r_raw, errors="coerce").reindex(bo.index)
            thrust += ((s50 > 30) & (s50.diff(3) > 2)).astype(float) * 0.25
        if bpspx_raw is not None:
            bp = pd.to_numeric(bpspx_raw, errors="coerce").reindex(bo.index)
            thrust += ((bp.diff(5) > 1.5) | (bp > bp.rolling(20, min_periods=10).mean())).astype(float) * 0.15
        return thrust.clip(0.0, 1.0)

def apply_snapshot_to_bundle(bundle: Dict[str, pd.Series], snapshot_map: Optional[Dict[str, float]]) -> Dict[str, pd.Series]:
    if not snapshot_map:
        return bundle
    updated = {k: v.copy() for k, v in bundle.items()}
    snap_date = pd.Timestamp.now().normalize()
    mapping = {
        "_spxa50r": ("$SPXA50R", "SPXA50R"),
        "_Bpspx": ("$BPSPX", "BPSPX"),
        "_Bpnya": ("$BPNYA", "BPNYA"),
        "_nymo": ("$NYMO", "NYMO"),
        "_nySI": ("$NYSI", "NYSI"),
        "_nyad": ("$NYAD", "NYAD"),
        "_trin": ("$TRIN", "TRIN"),
        "_cpc": ("$CPCE", "$CPC", "CPCE", "CPC"),
        "_cpce": ("$CPCE", "$CPC", "CPCE", "CPC"),
        "_oexa50r": ("$OEXA50R", "OEXA50R"),
        "_spxadp": ("SPXADP", "$SPXADP"),
        "rsp": ("RSP",),
        "URSP": ("URSP",),
        "spy": ("SPY",),
        "VXX": ("VXX",),
        "RSP_SPY": ("RSP:SPY", "RSP_SPY"),
        "IWM_SPY": ("IWM:SPY", "IWM_SPY"),
        "SMH_SPY": ("SMH:SPY", "SMH_SPY"),
        "XLF_SPY": ("XLF:SPY", "XLF_SPY"),
        "HYG_IEF": ("HYG:IEF", "HYG_IEF"),
        "SPXS_SVOL": ("SPXS:SVOL", "SPXS_SVOL"),
    }
    for stem, aliases in mapping.items():
        if stem in updated:
            val = snapshot_value(snapshot_map, *aliases)
            if val is not None:
                updated[stem].loc[snap_date] = float(val)
                updated[stem] = updated[stem].sort_index()
                updated[stem] = updated[stem][~updated[stem].index.duplicated(keep="last")]
    return updated

def build_oscillator(price: pd.Series, bundle: Optional[Dict[str, pd.Series]], cfg: AppConfig) -> pd.DataFrame:
    price = pd.to_numeric(price, errors="coerce").dropna().sort_index()
    price_curve = BellCurveTransform.calculate(price, cfg.lookback, cfg.smooth_price)
    
    if not bundle:
        df = pd.DataFrame({
            "price": price, "price_curve": price_curve, "breadth_curve": price_curve,
            "final_osc": price_curve, "quality_score": 0.50, "thrust_score": 0.00,
        })
        return df.dropna(subset=["price"])

    # Fuzzy matching helper
    def get_series(bundle_keys, possible_names):
        for name in possible_names:
            if name in bundle_keys:
                return bundle_keys[name]
        for key, series in bundle_keys.items():
            clean_key = key.replace("_", "").upper()
            for name in possible_names:
                clean_name = name.replace("_", "").upper()
                if clean_key == clean_name:
                    return series
        return None

    source_map = {
        "SPXA50R": get_series(bundle, ["_spxa50r", "SPXA50R", "$SPXA50R"]),
        "BPSPX": get_series(bundle, ["_Bpspx", "BPSPX", "$BPSPX"]),
        "BPNYA": get_series(bundle, ["_Bpnya", "BPNYA", "$BPNYA"]),
        "NYMO": get_series(bundle, ["_nymo", "NYMO", "$NYMO"]),
        "NYSI": get_series(bundle, ["_nySI", "NYSI", "$NYSI"]),
        "NYAD": get_series(bundle, ["_nyad", "NYAD", "$NYAD"]),
        "TRIN": get_series(bundle, ["_trin", "TRIN", "$TRIN"]),
        "CPCE": get_series(bundle, ["_cpce", "_cpc", "CPCE", "$CPCE"]),
        "OEXA50R": get_series(bundle, ["_oexa50r", "OEXA50R", "$OEXA50R"]),
        "SPXADP": get_series(bundle, ["_spxadp", "SPXADP", "$SPXADP"]),
    }
    
    transformed = {}
    raw_map = {}
    for name, series in source_map.items():
        if series is None or len(series) < 30:
            continue
        raw = pd.to_numeric(series, errors="coerce").sort_index()
        raw_map[name] = raw
        x = raw.copy()
        if name == "NYAD":
            x = x.pct_change(5)
        elif name in ("TRIN", "CPCE"):
            x = -x
        transformed[name] = BellCurveTransform.calculate(x, cfg.lookback, cfg.smooth_breadth)
        
    if not transformed:
        df = pd.DataFrame({
            "price": price, "price_curve": price_curve, "breadth_curve": price_curve,
            "final_osc": price_curve, "quality_score": 0.50, "thrust_score": 0.00,
        })
        return df.dropna(subset=["price"])
        
    breadth_df = pd.DataFrame(transformed).sort_index()
    weights = pd.Series(cfg.breadth_weights, dtype=float)
    cols = [c for c in breadth_df.columns if c in weights.index]
    
    # Dynamic weight normalization based on available data
    weighted_sum = breadth_df[cols].multiply(weights.loc[cols], axis=1).sum(axis=1, min_count=1)
    present_weights = breadth_df[cols].notna().multiply(weights.loc[cols], axis=1).sum(axis=1).replace(0, np.nan)
    breadth_curve = weighted_sum / present_weights
    
    thrust = BreadthThrustDetector.calculate(breadth_curve, raw_map.get("SPXA50R"), raw_map.get("BPSPX"))
    
    quality = pd.Series(0.0, index=breadth_curve.index)
    if "SPXA50R" in raw_map:
        s50 = raw_map["SPXA50R"].reindex(quality.index)
        quality += (s50 > 30).astype(float) * 0.40
        quality += (s50 > 40).astype(float) * 0.20
    if "BPSPX" in raw_map:
        bp = raw_map["BPSPX"].reindex(quality.index)
        quality += (bp > 40).astype(float) * 0.15
    if "NYMO" in raw_map:
        nymo = raw_map["NYMO"].reindex(quality.index)
        quality += (nymo > -10).astype(float) * 0.10
    quality += thrust.fillna(0.0) * 0.15
    quality = quality.clip(0.0, 1.0)
    
    final_osc = cfg.breadth_weight * breadth_curve + cfg.price_weight * price_curve
    
    df = pd.concat([
        price.rename("price"),
        price_curve.rename("price_curve"),
        breadth_curve.rename("breadth_curve"),
        final_osc.rename("final_osc"),
        quality.rename("quality_score"),
        thrust.rename("thrust_score"),
    ], axis=1)
    for col in cols:
        df[col] = breadth_df[col]
    return df.dropna(subset=["price"]).sort_index()

def run_backtest(df: pd.DataFrame, cfg: AppConfig) -> Dict[str, object]:
    work = df.dropna(subset=["price", "final_osc"]).copy()
    if len(work) < 100:
        return {"metrics": {}, "equity_curve": pd.Series(dtype=float), "position": pd.Series(dtype=float), "trades": pd.DataFrame()}
    
    detector = RegimeDetector()
    regime = detector.detect(work["price"])
    returns = work["price"].pct_change().fillna(0.0)
    vol = returns.rolling(20, min_periods=10).std() * np.sqrt(252)
    size = (cfg.target_vol / vol.clip(lower=0.05, upper=0.30)).clip(0.25, 1.00).fillna(0.50)
    
    position = pd.Series(0.0, index=work.index)
    strategy_returns = pd.Series(0.0, index=work.index)
    trades = []
    in_position = False
    entry_price = np.nan
    entry_date = None
    entry_osc = np.nan
    bars_held = 0
    
    cost_decimal = cfg.cost_bps / 10000.0
    
    for i in range(1, len(work)):
        dt = work.index[i]
        curr_price = float(work["price"].iloc[i])
        curr_osc = float(work["final_osc"].iloc[i])
        curr_quality = float(work["quality_score"].iloc[i])
        curr_thrust = float(work["thrust_score"].iloc[i])
        th = detector.thresholds(str(regime.iloc[i]))
        
        enter_long = (curr_osc <= th["long_enter"]) and (curr_quality >= cfg.min_quality_for_long or curr_thrust >= 0.60)
        exit_long = (curr_osc >= th["long_exit"]) or (curr_quality < 0.20 and curr_thrust < 0.20) or (bars_held >= cfg.max_hold_bars)
        
        stop_hit = False
        if in_position and pd.notna(entry_price):
            if (curr_price / entry_price - 1.0) <= -cfg.stop_loss_pct:
                stop_hit = True
                
        if not in_position and enter_long:
            in_position = True
            entry_price = curr_price
            entry_date = dt
            entry_osc = curr_osc
            bars_held = 0
            strategy_returns.iloc[i] -= cost_decimal
        elif in_position and (exit_long or stop_hit):
            exit_price = curr_price
            trades.append({
                "entry_date": entry_date, "exit_date": dt, "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2),
                "pnl": exit_price / entry_price - 1.0, "bars_held": bars_held, "entry_osc": round(entry_osc, 2),
                "exit_osc": round(curr_osc, 2), "exit_reason": "stop_loss" if stop_hit else "signal", "regime": str(regime.iloc[i]),
            })
            in_position = False
            entry_price = np.nan
            entry_date = None
            entry_osc = np.nan
            bars_held = 0
            strategy_returns.iloc[i] -= cost_decimal
            
        position.iloc[i] = 1.0 if in_position else 0.0
        strategy_returns.iloc[i] += returns.iloc[i] * position.iloc[i] * float(size.iloc[i])
        if in_position:
            bars_held += 1
            
    if in_position and entry_date is not None and pd.notna(entry_price):
        exit_price = float(work["price"].iloc[-1])
        exit_osc = float(work["final_osc"].iloc[-1])
        trades.append({
            "entry_date": entry_date, "exit_date": work.index[-1], "entry_price": round(entry_price, 2), "exit_price": round(exit_price, 2),
            "pnl": exit_price / entry_price - 1.0, "bars_held": bars_held, "entry_osc": round(entry_osc, 2),
            "exit_osc": round(exit_osc, 2), "exit_reason": "end_of_data", "regime": str(regime.iloc[-1]),
        })
        
    equity = (1.0 + strategy_returns.fillna(0.0)).cumprod()
    trades_df = pd.DataFrame(trades)
    metrics = {}
    
    if not equity.empty:
        years = max(len(strategy_returns.dropna()) / 252.0, 1 / 252.0)
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else np.nan
        vol_ann = float(strategy_returns.std() * np.sqrt(252))
        sharpe = float(cagr / vol_ann) if vol_ann > 0 and pd.notna(cagr) else np.nan
        drawdown = equity / equity.cummax() - 1.0
        max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
        win_rate = float((trades_df["pnl"] > 0).mean()) if not trades_df.empty else np.nan
        bh_total = float(work["price"].iloc[-1] / work["price"].iloc[0] - 1.0)
        bh_cagr = float((1.0 + bh_total) ** (1.0 / years) - 1.0) if bh_total > -1 else np.nan
        
        metrics = {
            "total_return": total_return, "cagr": cagr, "volatility": vol_ann, "sharpe": sharpe,
            "max_drawdown": max_dd, "win_rate": win_rate, "num_trades": int(len(trades_df)),
            "bh_cagr": bh_cagr, "outperformance": cagr - bh_cagr if pd.notna(cagr) and pd.notna(bh_cagr) else np.nan,
        }
    return {"metrics": metrics, "equity_curve": equity, "position": position, "trades": trades_df, "strategy_returns": strategy_returns}

def summarize_signal(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {}
    detector = RegimeDetector()
    regime = detector.detect(df["price"])
    last_idx = df.index[-1]
    last_regime = str(regime.loc[last_idx])
    th = detector.thresholds(last_regime)
    osc = float(df["final_osc"].iloc[-1])
    quality = float(df["quality_score"].iloc[-1])
    thrust = float(df["thrust_score"].iloc[-1])
    
    if osc <= th["long_enter"] and (quality >= 0.45 or thrust >= 0.60):
        state = "LONG"
    elif osc >= th["risk_off"]:
        state = "RISK-OFF / TAKE PROFITS"
    else:
        state = "HOLD / NEUTRAL"
        
    return {
        "date": last_idx, "state": state, "regime": last_regime, "final_osc": osc, "quality_score": quality,
        "thrust_score": thrust, "long_enter": th["long_enter"], "long_exit": th["long_exit"], "risk_off": th["risk_off"],
    }

@st.cache_data(show_spinner=False)
def build_model(zip_bytes, snapshot_bytes, ticker, period, lookback, smooth_price, smooth_breadth, 
                breadth_weight, stop_loss_pct, target_vol, max_hold_bars, cost_bps):
    
    snapshot_map = parse_snapshot_csv(snapshot_bytes) if snapshot_bytes is not None else None
    bundle = load_zip_bundle(zip_bytes) if zip_bytes is not None else {}
    if bundle:
        bundle = apply_snapshot_to_bundle(bundle, snapshot_map)
        
    price = None
    for stem in [ticker, ticker.lower(), ticker.upper()]:
        if bundle and stem in bundle:
            price = bundle[stem].rename(ticker.upper())
            break
    if price is None or price.empty:
        price = fetch_price(ticker, period)
        
    cfg = AppConfig(
        lookback=lookback, smooth_price=smooth_price, smooth_breadth=smooth_breadth,
        breadth_weight=breadth_weight, price_weight=1.0 - breadth_weight,
        stop_loss_pct=stop_loss_pct, target_vol=target_vol, max_hold_bars=max_hold_bars,
        cost_bps=cost_bps
    )
    osc_df = build_oscillator(price, bundle, cfg)
    backtest = run_backtest(osc_df, cfg)
    summary = summarize_signal(osc_df)
    return osc_df, backtest, summary, bundle, snapshot_map

def trim_df(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    if bars == 0 or len(df) <= bars:
        return df.copy()
    return df.iloc[-bars:].copy()

def bars_from_label(label: str) -> int:
    return {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "10Y": 2520, "Max": 0}.get(label, 252)

def oscillator_chart(df: pd.DataFrame, bars: int, view: str) -> go.Figure:
    plot_df = trim_df(df, bars)
    fig = go.Figure()
    if view in ("Combined", "All"):
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["final_osc"], mode="lines", name="Final Osc", line=dict(width=3)))
    if view in ("Breadth", "All"):
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["breadth_curve"], mode="lines", name="Breadth Curve", line=dict(width=2, dash="dot")))
    if view in ("Price", "All"):
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["price_curve"], mode="lines", name="Price Curve", line=dict(width=2, dash="dash")))
    
    fig.add_hrect(y0=0, y1=20, opacity=0.12, line_width=0)
    fig.add_hrect(y0=20, y1=40, opacity=0.08, line_width=0)
    fig.add_hrect(y0=40, y1=60, opacity=0.05, line_width=0)
    fig.add_hrect(y0=60, y1=80, opacity=0.08, line_width=0)
    fig.add_hrect(y0=80, y1=100, opacity=0.12, line_width=0)
    if not plot_df.empty:
        fig.add_trace(go.Scatter(x=[plot_df.index[-1]], y=[plot_df["final_osc"].iloc[-1]], mode="markers", name="Current", marker=dict(size=12, symbol="diamond")))
    
    fig.update_layout(title="Ultimate Oscillator", height=520, yaxis_title="0–100 Score", yaxis=dict(range=[0, 100]), xaxis_title="Date", hovermode="x unified", legend=dict(orientation="h"), margin=dict(l=40, r=20, t=60, b=40))
    return fig

def price_chart(df: pd.DataFrame, bars: int) -> go.Figure:
    plot_df = trim_df(df, bars)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["price"], mode="lines", name="Price", line=dict(width=2)))
    fig.update_layout(title="Price", height=320, xaxis_title="Date", yaxis_title="Price", hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
    return fig

def equity_chart(backtest: Dict[str, object]) -> go.Figure:
    eq = backtest.get("equity_curve", pd.Series(dtype=float))
    fig = go.Figure()
    if isinstance(eq, pd.Series) and not eq.empty:
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Strategy Equity", line=dict(width=2)))
    fig.update_layout(title="Backtest Equity Curve", height=320, xaxis_title="Date", yaxis_title="Equity", hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
    return fig

# --- UI ---

st.title("GLM Ultimate Oscillator Pro")
st.caption("Robust backtesting with fuzzy file matching, regime detection, and configurable costs.")

with st.sidebar:
    st.header("Data")
    breadth_zip = st.file_uploader("Historical breadth ZIP", type=["zip"])
    snapshot_csv = st.file_uploader("Daily snapshot CSV", type=["csv"])
    ticker = st.selectbox("Primary ticker", ["RSP", "URSP", "SPY"], index=0)
    period = st.selectbox("Yahoo fallback history", ["2y", "5y", "10y", "max"], index=2)
    
    st.header("Model")
    lookback = st.slider("Bell curve lookback", 63, 252, 126, step=21)
    smooth_price = st.slider("Price smoothing", 0.0, 8.0, 3.5, step=0.5)
    smooth_breadth = st.slider("Breadth smoothing", 0.0, 8.0, 3.0, step=0.5)
    breadth_weight = st.slider("Breadth weight", 0.0, 1.0, 0.70, step=0.05)
    
    st.header("Risk / Timing")
    stop_loss_pct = st.slider("Stop loss %", 0.02, 0.12, 0.06, step=0.01)
    target_vol = st.slider("Target volatility", 0.05, 0.25, 0.12, step=0.01)
    max_hold_bars = st.slider("Max hold bars", 5, 40, 20, step=1)
    # FIX: Added Transaction Cost Slider
    cost_bps = st.slider("Transaction Cost (bps)", 1, 10, 3, step=1)
    
    st.header("View")
    view_mode = st.selectbox("Oscillator view", ["Combined", "Breadth", "Price", "All"], index=0)
    chart_window = st.selectbox("Chart window", ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "Max"], index=2)
    run = st.button("Build Ultimate Oscillator", type="primary", use_container_width=True)

st.info("✅ **Ready.** Upload your files or rely on Yahoo Finance fallback.")

if not run:
    st.stop()

try:
    zip_bytes = breadth_zip.getvalue() if breadth_zip is not None else None
    snapshot_bytes = snapshot_csv.getvalue() if snapshot_csv is not None else None
    with st.spinner("Building model..."):
        osc_df, backtest, summary, bundle, snapshot_map = build_model(
            zip_bytes=zip_bytes, snapshot_bytes=snapshot_bytes, ticker=ticker, period=period,
            lookback=lookback, smooth_price=smooth_price, smooth_breadth=smooth_breadth,
            breadth_weight=breadth_weight, stop_loss_pct=stop_loss_pct, target_vol=target_vol, 
            max_hold_bars=max_hold_bars, cost_bps=cost_bps
        )
    
    if osc_df.empty:
        st.error("No model output was produced. Check your file formats or ticker data.")
        st.stop()
        
    bars = bars_from_label(chart_window)
    
    # Summary Metrics
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("State", str(summary.get("state", "N/A")))
    top2.metric("Regime", str(summary.get("regime", "N/A")))
    top3.metric("Final Osc", f"{summary.get('final_osc', float('nan')):.2f}")
    top4.metric("Quality", f"{summary.get('quality_score', float('nan')):.2f}")
    
    mid1, mid2, mid3 = st.columns(3)
    mid1.metric("Thrust", f"{summary.get('thrust_score', float('nan')):.2f}")
    mid2.metric("Long Enter", f"{summary.get('long_enter', float('nan')):.2f}")
    mid3.metric("Long Exit", f"{summary.get('long_exit', float('nan')):.2f}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Backtest", "Data", "Inputs"])
    
    with tab1:
        st.plotly_chart(oscillator_chart(osc_df, bars, view_mode), use_container_width=True)
        st.plotly_chart(price_chart(osc_df, bars), use_container_width=True)
        
    with tab2:
        metrics = backtest.get("metrics", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Return", f"{metrics.get('total_return', float('nan')):.2%}" if metrics else "N/A")
        m2.metric("CAGR", f"{metrics.get('cagr', float('nan')):.2%}" if metrics else "N/A")
        m3.metric("Sharpe", f"{metrics.get('sharpe', float('nan')):.2f}" if metrics else "N/A")
        m4.metric("Max DD", f"{metrics.get('max_drawdown', float('nan')):.2%}" if metrics else "N/A")
        m5.metric("Trades", str(metrics.get("num_trades", 0)) if metrics else "0")
        
        n1, n2 = st.columns(2)
        n1.metric("Buy & Hold CAGR", f"{metrics.get('bh_cagr', float('nan')):.2%}" if metrics else "N/A")
        n2.metric("Outperformance", f"{metrics.get('outperformance', float('nan')):.2%}" if metrics else "N/A")
        
        st.plotly_chart(equity_chart(backtest), use_container_width=True)
        
        trades = backtest.get("trades", pd.DataFrame())
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            show_trades = trades.copy()
            show_trades["pnl"] = (show_trades["pnl"] * 100).round(2)
            st.dataframe(show_trades.tail(50), use_container_width=True)
        else:
            st.caption("No trades generated with current settings.")
            
    with tab3:
        st.subheader("Latest rows")
        st.dataframe(osc_df.tail(50), use_container_width=True)
        st.subheader("Loaded series from zip")
        loaded = sorted(bundle.keys()) if bundle else []
        st.write(loaded if loaded else "No zip series loaded.")
        
    with tab4:
        st.subheader("Snapshot symbols")
        if snapshot_map:
            preview = pd.DataFrame({"Symbol": list(snapshot_map.keys()), "Close": list(snapshot_map.values())}).sort_values("Symbol")
            st.dataframe(preview, use_container_width=True, hide_index=True)
        else:
            st.write("No snapshot CSV loaded.")
        st.markdown("""
**Expected files**
- Historical breadth ZIP (supports both CSV and Pipe-delimited)
- Daily snapshot CSV

**Primary signals**
- Breadth dominates price by default
- Long bias when oscillator is washed out and quality/thrust improve
- Risk-off when oscillator is stretched high
""")

except Exception as e:
    st.error(f"App failed: {e}")
    st.code(traceback.format_exc())
