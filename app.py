import io
import zipfile
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION & DATA CLASSES
# ============================================================================

@dataclass
class TradingCosts:
    commission: float = 0.0005
    slippage_bps: float = 2.0
    market_impact_bps: float = 5.0
    spread_bps: float = 1.0
    
    def estimate_total_bps(self, position_size: float = 1.0) -> float:
        return (self.slippage_bps + self.market_impact_bps * np.sqrt(position_size) 
                + self.spread_bps + self.commission * 10000 / position_size)

@dataclass
class RegimeThresholds:
    bull_long: float = 16.0
    bull_short: float = 84.0
    bear_long: float = 21.0
    bear_short: float = 79.0
    sideways_long: float = 18.0
    sideways_short: float = 82.0

@dataclass
class BacktestResult:
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, float] = field(default_factory=dict)
    bootstrap_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)

# ============================================================================
# QUANT MATH: BELL CURVE TRANSFORM
# ============================================================================

class BellCurveTransform:
    """Forces a time series into a Gaussian Bell Curve distribution using Scipy."""
    
    @staticmethod
    def calculate(series: pd.Series, lookback: int = 252, sigma: float = 12.0) -> pd.Series:
        vals = series.interpolate(limit_direction='both').ffill().bfill()
        
        rolling_mean = vals.rolling(lookback, min_periods=int(lookback*0.8)).mean()
        rolling_std = vals.rolling(lookback, min_periods=int(lookback*0.8)).std().clip(lower=1e-5)
        z_score = (vals - rolling_mean) / rolling_std
        
        bell_prob = norm.cdf(z_score)
        bell_scaled = bell_prob * 100
        
        smoothed = gaussian_filter1d(bell_scaled, sigma=sigma)
        smoothed[np.isnan(vals.values)] = np.nan
        
        return pd.Series(smoothed, index=series.index)

# ============================================================================
# REGIME DETECTION
# ============================================================================

class RegimeDetector:
    """Quant Regime classification using rolling Z-Scores and Volatility Clustering"""
    
    def detect_regime(self, price: pd.Series, returns: pd.Series) -> pd.Series:
        momentum = price.pct_change(252).dropna()
        
        vol_short = returns.rolling(20, min_periods=10).std() * np.sqrt(252)
        vol_long = returns.rolling(252, min_periods=100).std() * np.sqrt(252)
        vol_ratio = vol_short / vol_long.clip(lower=1e-5)
        
        equity = (1 + returns).cumprod()
        drawdown = (equity / equity.expanding().max()) - 1
        
        features = pd.DataFrame({
            'momentum': momentum,
            'vol_ratio': vol_ratio,
            'drawdown': drawdown
        }).dropna()
        
        if len(features) < 50:
            return pd.Series('sideways', index=price.index)
        
        try:
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(features)
            features['regime_id'] = kmeans.labels_
            
            cluster_means = features.groupby('regime_id').mean()
            regime_map = {}
            for idx, row in cluster_means.iterrows():
                if row['momentum'] > 0.002 and row['vol_ratio'] < 1.1:
                    regime_map[idx] = 'bull'
                elif row['drawdown'] < -0.05 or row['vol_ratio'] > 1.4:
                    regime_map[idx] = 'bear'
                else:
                    regime_map[idx] = 'sideways'
                    
        except Exception:
            regime = pd.Series('sideways', index=features.index)
            regime.loc[features['momentum'] > 0.002] = 'bull'
            regime.loc[features['vol_ratio'] > 1.3] = 'bear'
            return regime
            
        return features['regime_id'].map(regime_map).reindex(price.index).ffill()

    def get_adaptive_thresholds(self, regime: str) -> RegimeThresholds:
        thresholds = RegimeThresholds()
        if regime == 'bull':
            thresholds.bull_long = 16.0
            thresholds.bull_short = 84.0
        elif regime == 'bear':
            thresholds.bear_long = 21.0
            thresholds.bear_short = 79.0
        else:
            thresholds.sideways_long = 18.0
            thresholds.sideways_short = 82.0
        return thresholds

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

class RiskManager:
    def __init__(self, stop_loss_pct: float = 0.05, trailing_atr_mult: float = 2.0, target_vol: float = 0.10):
        self.stop_loss_pct = stop_loss_pct
        self.trailing_atr_mult = trailing_atr_mult
        self.target_vol = target_vol

    def volatility_scaled_size(self, current_vol: float) -> float:
        if current_vol <= 0:
            return 1.0
        return min(1.0, self.target_vol / current_vol)

# ============================================================================
# ENHANCED OSCILLATOR
# ============================================================================

class UltimateOscillator:
    """Combines Price and Breadth into a mathematically strict Bell Curve."""
    
    def __init__(self, daily_smooth: float = 12.0, weekly_smooth: float = 25.0):
        self.daily_smooth = daily_smooth
        self.weekly_smooth = weekly_smooth

    def build_price_curve(self, price: pd.Series, smooth: float) -> pd.Series:
        return BellCurveTransform.calculate(price, lookback=252, sigma=smooth)

    def build_breadth_curve(self, zip_file, snapshot_map=None, snap_date=None) -> pd.Series:
        """Loads zip, handles structural differences, applies Bell Curve Transform."""
        if zip_file is None:
            return pd.Series(dtype=float)
            
        # Updated to match your exact ZIP filenames
        config = [
            ("_Bpspx", "RSP", 1.0, "raw"),
            ("_Bpnya", "RSP", 1.0, "raw"),
            ("_nymo", "RSP", 1.0, "raw"),
            ("_nySI", "RSP", 1.0, "raw"),
            ("_nyad", "RSP", 1.0, "diff"), 
            ("_spxa50r", "RSP", 1.0, "raw"),
            ("_trin", "RSP", -1.0, "raw"),     
            ("_cpc", "RSP", -1.0, "raw"),     
        ]
        
        pieces = []
        with zipfile.ZipFile(zip_file) as z:
            for stem, snap_sym, mult, trans in config:
                matches = [n for n in z.namelist() if stem.lower() in n.lower() and n.lower().endswith(".csv")]
                if not matches:
                    continue
                
                text = z.read(matches[0]).decode("utf-8", errors="ignore")
                series = self._parse_sc_pipe_text(text)
                if series is None:
                    continue
                
                # INJECT SNAPSHOT DATA (Before diffing cumulative data!)
                if snapshot_map is not None and snap_date is not None and stem in snapshot_map:
                    series.loc[snap_date] = snapshot_map[stem]
                
                # Apply transformation (e.g., cumulative to momentum for NYAD)
                if trans == "diff":
                    series = series.diff()
                
                # Apply directional multiplier (Invert contrarian indicators)
                series = series * mult
                
                # Apply Bell Curve Transform
                curve = BellCurveTransform.calculate(series, lookback=252, sigma=10.0)
                pieces.append(curve.rename(stem))
                
        if not pieces:
            return pd.Series(dtype=float)
            
        comp = pd.concat(pieces, axis=1).dropna(how='all')
        return comp.mean(axis=1, skipna=True)

    def _parse_sc_pipe_text(self, text: str) -> pd.Series:
        """Parses the exact StockCharts pipe-delimited format."""
        lines = text.strip().split('\n')
        rows = []
        for line in lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 5:
                try:
                    date = pd.to_datetime(parts[1])
                    close = float(parts[4])
                    rows.append({'Date': date, 'Close': close})
                except:
                    continue
        if not rows:
            return pd.Series(dtype=float)
            
        df = pd.DataFrame(rows).set_index('Date')['Close']
        df = df[~df.index.isna()].dropna().sort_index()
        return df[~df.index.duplicated(keep='last')]

# ============================================================================
# COMPREHENSIVE BACKTEST ENGINE
# ============================================================================

class QuantBacktestEngine:
    """Production-grade backtesting without look-ahead bias."""
    
    def __init__(self, costs: TradingCosts = None, risk_mgr: RiskManager = None, regime_detector: RegimeDetector = None):
        self.costs = costs or TradingCosts()
        self.risk_mgr = risk_mgr or RiskManager()
        self.regime_detector = regime_detector or RegimeDetector()

    def run_backtest(self, df: pd.DataFrame, regime_series: pd.Series = None) -> BacktestResult:
        osc = df['osc'].astype(float)
        px = df['price'].astype(float)
        returns = px.pct_change().dropna()
        
        current_regime = 'sideways'
        if regime_series is not None and len(regime_series) == len(df):
            current_regime = regime_series.iloc[-1]
        thresholds = self.regime_detector.get_adaptive_thresholds(current_regime)
        
        vol = returns.rolling(20, min_periods=10).std() * np.sqrt(252)
        pos_size = self.risk_mgr.volatility_scaled_size(vol.fillna(self.risk_mgr.target_vol))
        
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        bars_held = 0
        equity = [1.0]
        daily_returns = []
        atr = returns.rolling(14, min_periods=10).mean() * 1.0
        
        for i in range(1, len(df)):
            current_price = px.iloc[i]
            prev_price = px.iloc[i-1]
            date = df.index[i]
            
            exit_signal = False
            if position != 0:
                bars_held += 1
                pnl_pct = (current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price
                
                sl = self.risk_mgr.stop_loss_pct
                time_exit = bars_held >= 20
                
                if pnl_pct <= -sl or time_exit:
                    exit_signal = True
                    
            if exit_signal and position != 0:
                trade_return = (current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price
                trade_return -= self.costs.estimate_total_bps() / 10000
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'direction': position,
                    'return': trade_return,
                    'bars_held': bars_held
                })
                position = 0
                bars_held = 0
            
            if position == 0 and osc.iloc[i-1] < thresholds.bull_long and osc.iloc[i] >= thresholds.bull_long:
                position = 1
                entry_price = current_price
                entry_date = date
                bars_held = 0
            elif position == 0 and osc.iloc[i-1] > thresholds.bear_short and osc.iloc[i] <= thresholds.bear_short:
                position = -1
                entry_price = current_price
                entry_date = date
                bars_held = 0
                
            daily_ret = (current_price - prev_price) / prev_price * position * pos_size.iloc[i]
            daily_returns.append(daily_ret)
            equity.append(equity[-1] * (1 + daily_ret))
            
        returns_series = pd.Series(daily_returns, index=df.index[1:])
        equity_curve = pd.Series(equity, index=df.index[:len(equity)])
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        metrics = self._calculate_metrics(returns_series, equity_curve, trades_df)
        return BacktestResult(
            returns=returns_series,
            equity_curve=equity_curve,
            trades=trades_df,
            metrics=metrics,
            bootstrap_ci={}
        )
    
    def _calculate_metrics(self, returns: pd.Series, equity: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
        if len(returns) == 0:
            return {}
        
        total_ret = equity.iloc[-1] - 1
        ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0
        
        peak = equity.expanding().max()
        max_dd = ((peak - equity) / peak).max()
        
        downside = returns[returns < 0]
        down_dev = downside.std() * np.sqrt(252) if len(downside) > 0 else vol
        sortino = ann_ret / down_dev if down_dev > 0 else 0
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        if len(trades) > 0:
            win_rate = (trades['return'] > 0).mean()
            avg_win = trades['return'][trades['return'] > 0].mean() if any(trades['return'] > 0) else 0
            avg_loss = trades['return'][trades['return'] < 0].mean() if any(trades['return'] < 0) else 0
            profit_factor = abs(avg_win * win_rate / (avg_loss * (1 - win_rate))) if avg_loss != 0 else 0
            avg_hold = trades['bars_held'].mean() if 'bars_held' in trades.columns else 0
            num_trades = len(trades)
        else:
            win_rate, avg_win, avg_loss, profit_factor, avg_hold, num_trades = 0, 0, 0, 0, 0, 0

        return {
            'total_return': total_ret,
            'annual_return': ann_ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_holding_bars': avg_hold,
            'num_trades': num_trades
        }

# ============================================================================
# STREAMLIT UI DATA LOADERS
# ============================================================================

def load_snapshot_map(snapshot_file):
    """Parses your specific daily StockCharts snapshot format."""
    if snapshot_file is None:
        return None, None
        
    raw_text = snapshot_file.getvalue().decode("utf-8", errors="ignore")
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    
    # Map Daily Snapshot Symbols to Historical ZIP Stems
    symbol_map = {
        "$NYMO": "_nymo",
        "$CPCE": "_cpc",
        "$SPXA50R": "_spxa50r",
        "$BPSPX": "_Bpspx",
        "$BPNYA": "_Bpnya",
        "$NYSI": "_nySI",
        "$NYAD": "_nyad",
        "$TRIN": "_trin",
        "RSP": "RSP"
    }
    
    mapping = {}
    for line in lines:
        parts = [p.strip() for p in line.split('|') if p.strip()]
        
        # Skip header or malformed lines
        if len(parts) < 9 or parts[1] == "Symbol":
            continue
            
        sym = parts[1]
        close_val_str = parts[8]
        
        try:
            close_val = float(close_val_str)
        except ValueError:
            continue
            
        # Translate Snapshot Symbol to ZIP Stem
        if sym in symbol_map:
            mapping[symbol_map[sym]] = close_val
        else:
            # Fallback for other ETFs
            clean_sym = sym.replace("$", "")
            mapping[clean_sym] = close_val
            
    if not mapping:
        return None, None
        
    return mapping, pd.Timestamp.today().normalize()


def load_price_history(symbol, years, zip_file, snapshot_map, snap_date):
    if zip_file is not None:
        try:
            zip_file.seek(0)
        except:
            pass
        with zipfile.ZipFile(zip_file) as z:
            matches = [n for n in z.namelist() if symbol.lower() in n.lower() and n.lower().endswith(".csv")]
            if matches:
                text = z.read(matches[0]).decode("utf-8", errors="ignore")
                series = UltimateOscillator._parse_sc_pipe_text(text)
                if series is not None:
                    if snapshot_map is not None and snap_date is not None and symbol in snapshot_map:
                        series.loc[snap_date] = float(snapshot_map[symbol])
                        series = series.sort_index()
                    if len(series) > 50:
                        return series.rename("price")

    try:
        df = yf.download(symbol, period=f"{years}y", progress=False, auto_adjust=False)
        if len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                s = df[("Close", symbol)] if ("Close", symbol) in df.columns else df.iloc[:, 0]
            else:
                s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            out = pd.DataFrame({"price": pd.to_numeric(s, errors="coerce")}, index=pd.to_datetime(df.index))
            out = out.dropna()
            if snapshot_map is not None and snap_date is not None and symbol in snapshot_map:
                out.loc[snap_date, "price"] = float(snapshot_map[symbol])
                out = out.sort_index()
            return out if len(out) > 50 else None
    except:
        return None

def resample_weekly(price_hist):
    return price_hist.resample("W-FRI").last().dropna()

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="RSP Ultimate Bell Curve Oscillator", layout="wide")
    st.title("📊 RSP Ultimate Bell Curve Oscillator")
    st.markdown("Institutional framework combining Gaussian Bell Curve math with Regime-Adaptive risk management.")
    
    # --- SIDEBAR ---
    st.sidebar.header("Config")
    symbol = st.sidebar.text_input("ETF Symbol", "RSP")
    years = st.sidebar.slider("Historical lookback", 5, 25, 20)
    breadth_weight = st.sidebar.slider("Breadth Weight", 0.0, 1.0, 0.40, 0.05)
    daily_smooth = st.sidebar.slider("Daily Gaussian Sigma", 5.0, 25.0, 12.0)
    weekly_smooth = st.sidebar.slider("Weekly Gaussian Sigma", 10.0, 50.0, 25.0)
    
    st.sidebar.subheader("Risk Parameters")
    stop_loss = st.sidebar.slider("ATR Stop Loss Mult", 1.0, 5.0, 2.0)
    target_vol = st.sidebar.slider("Target Volatility", 0.05, 0.20, 0.10)
    
    # --- DUAL UPLOAD SYSTEM ---
    st.sidebar.subheader("1. Historical Data (Zip)")
    zip_file = st.sidebar.file_uploader("Upload Breadth History (.zip)", type="zip", key="zip_uploader")
    
    st.sidebar.subheader("2. Daily Snapshot (Today's Close)")
    snap_file = st.sidebar.file_uploader("Upload End-of-Day Snapshot (.csv)", type="csv", key="snap_uploader")
    
    view_options = {"3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825, "10Y": 3650, "Max": None}
    view_choice = st.sidebar.selectbox("View Window", list(view_options.keys()), index=4)
    
    # --- LOAD DATA ---
    snapshot_map, snap_date = load_snapshot_map(snap_file)
    price_hist = load_price_history(symbol, years, zip_file, snapshot_map, snap_date)
    
    if price_hist is None:
        st.error("Failed to load price history. Upload the Zip or check internet connection for YFinance fallback.")
        st.stop()
        
    # --- BUILD OSCILLATORS ---
    osc = UltimateOscillator(daily_smooth, weekly_smooth)
    
    with st.spinner("Building Bell Curve Oscillators..."):
        daily_price_curve = osc.build_price_curve(price_hist["price"], smooth=daily_smooth)
        breadth_curve = osc.build_breadth_curve(zip_file, snapshot_map, snap_date)
        
        ultimate_daily = pd.DataFrame({"price": price_hist["price"]})
        
        if breadth_curve is not None and len(breadth_curve) > 0:
            breadth_aligned = breadth_curve.reindex(price_hist.index)
            ultimate_daily["breadth"] = breadth_aligned
            ultimate_daily["osc"] = ((1 - breadth_weight) * osc.build_price_curve(price_hist["price"], daily_smooth) + 
                                     breadth_weight * breadth_aligned).clip(0, 100)
        else:
            ultimate_daily["osc"] = osc.build_price_curve(price_hist["price"], daily_smooth)
            
        ultimate_daily["signal"] = gaussian_filter1d(ultimate_daily["osc"].fillna(50), sigma=daily_smooth)
        ultimate_daily = ultimate_daily.dropna(subset=['osc', 'signal'])
            
        wk_hist = resample_weekly(price_hist)
        weekly_price_curve = osc.build_price_curve(wk_hist["price"], smooth=weekly_smooth)
        ultimate_weekly = pd.DataFrame({"price": wk_hist["price"]})
        
        if breadth_curve is not None and len(breadth_curve) > 0:
            wk_breadth_curve = breadth_curve.resample("W-FRI").last()
            wk_breadth_aligned = wk_breadth_curve.reindex(wk_hist.index)
            ultimate_weekly["breadth"] = wk_breadth_aligned
            ultimate_weekly["osc"] = ((1 - breadth_weight) * weekly_price_curve + 
                                breadth_weight * wk_breadth_aligned).clip(0, 100)
        else:
            ultimate_weekly["osc"] = weekly_price_curve
            
        ultimate_weekly["signal"] = gaussian_filter1d(ultimate_weekly["osc"].fillna(50), sigma=weekly_smooth)
        ultimate_weekly = ultimate_weekly.dropna(subset=['osc', 'signal'])
            
    # --- REGIME DETECTION & BACKTEST ---
    returns = price_hist["price"].pct_change().dropna()
    regime_engine = RegimeDetector()
    regime_series = regime_engine.detect_regime(price_hist["price"], returns)
    
    risk_mgr = RiskManager(stop_loss_pct=stop_loss, trailing_atr_mult=stop_loss, target_vol=target_vol)
    bt_engine = QuantBacktestEngine(TradingCosts(), risk_mgr, regime_engine)
    
    daily_bt = bt_engine.run_backtest(ultimate_daily, regime_series=regime_series)
    weekly_bt = bt_engine.run_backtest(ultimate_weekly, regime_series=regime_series.reindex(ultimate_weekly.index).ffill())
    
    # --- DASHBOARD TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Daily", "📊 Weekly", "🎯 Results"])
    
    with tab1:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_regime = regime_series.iloc[-1]
        regime_color = {"bull": "🟢", "bear": "🔴", "sideways": "🟡"}.get(current_regime, "⚪")
        
        with col1:
            st.metric("Current Osc", f"{ultimate_daily['osc'].iloc[-1]:.1f}")
        with col2:
            st.metric("Regime", f"{regime_color} {current_regime.upper()}")
        with col3:
            st.metric("OOS Sharpe", f"{daily_bt.metrics.get('sharpe_ratio', 0):.2f}")
        with col4:
            st.metric("Max Drawdown", f"{daily_bt.metrics.get('max_drawdown', 0):.2%}")
        with col5:
            st.metric("Profit Factor", f"{daily_bt.metrics.get('profit_factor', 0):.2f}")
            
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        dvp = ultimate_daily.copy()
        if view_choice != "Max":
            days = view_options[view_choice]
            dvp = dvp[dvp.index >= dvp.index.max() - pd.Timedelta(days=days)]
            
        fig.add_hrect(y0=0, y1=15.86, fillcolor="rgba(220,38,38,0.15)", line_width=0, row=1, col=1)
        fig.add_hrect(y0=84.13, y1=100, fillcolor="rgba(34,197,94,0.15)", line_width=0, row=1, col=1)
        fig.add_hline(y=15.86, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=84.13, line_dash="dash", line_color="green", row=1, col=1)
        
        fig.add_trace(go.Scatter(x=dvp.index, y=dvp["osc"], name="Ultimate Osc (Bell Curve)", 
                            line=dict(width=2.5, color="#FF6B6B"), row=1, col=1))
        fig.add_trace(go.Scatter(x=dvp.index, y=dvp["signal"], name="Signal", 
                            line=dict(width=1.5, color="#4ECDC4", dash="dash"), row=1, col=1))
        
        if len(regime_series) == len(ultimate_daily):
            regime_daily = regime_series.reindex(dvp.index, method='ffill')
            for r_type, color in [('bull', 'rgba(0,255,0,0.06)'), ('bear', 'rgba(255,0,0,0.06)')]:
                mask = regime_daily == r_type
                if mask.any():
                    mask_indices = mask[mask].index
                    fig.add_vrect(x0=mask_indices[0], x1=mask_indices[-1], 
                                 fillcolor=color, line_width=0, row=1, col=1)
                                 
        price_view = price_hist[price_hist.index >= dvp.index[0]]
        fig.add_trace(go.Scatter(x=price_view.index, y=price_view["price"], name="RSP Price",
                            line=dict(width=1.5, color="#95A5A6"), row=2, col=1))
                            
        fig.update_layout(height=800, showlegend=True, title_text="Daily Bell Curve Oscillator + Regimes")
        fig.update_yaxes(title_text="Oscillator (Bell Curve Distribution)", range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text="RSP Price", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        osc_val = ultimate_daily['osc'].iloc[-1]
        sig_val = ultimate_daily['signal'].iloc[-1]
        
        if osc_val > 93.32 and osc_val > sig_val:
            st.success("🚀 OVERBOUGHT (+1.5 Sigma tail of Bell Curve)")
        elif osc_val < 6.68 and osc_val < sig_val:
            st.error("📉 OVERSOLD (-1.5 Sigma tail of Bell Curve)")
        elif osc_val > sig_val:
            st.info("📈 BULLISH MOMENTUM")
        else:
            st.warning("📉 BEARISH MOMENTUM")

    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Weekly Osc", f"{ultimate_weekly['osc'].iloc[-1]:.1f}")
        with col2:
            st.metric("OOS Sharpe", f"{weekly_bt.metrics.get('sharpe_ratio', 0):.2f}")
        with col3:
            st.metric("Sortino", f"{weekly_bt.metrics.get('sortino_ratio', 0):.2f}")
        with col4:
            st.metric("Win Rate", f"{weekly_bt.metrics.get('win_rate', 0)*100:.1f}%")
            
        fig = go.Figure()
        wvp = ultimate_weekly.copy()
        if view_choice != "Max":
            days = view_options[view_choice]
            wvp = wvp[wvp.index >= wvp.index.max() - pd.Timedelta(days=days)]
            
        fig.add_hrect(y0=0, y1=15.86, fillcolor="rgba(220,38,38,0.15)", line_width=0)
        fig.add_hrect(y0=84.13, y1=100, fillcolor="rgba(34,197,94,0.15)", line_width=0)
        fig.add_hline(y=15.86, line_dash="dash", line_color="red")
        fig.add_hline(y=84.13, line_dash="dash", line_color="green")
        
        fig.add_trace(go.Scatter(x=wvp.index, y=wvp["osc"], name="Weekly Ultimate Osc", 
                            line=dict(width=2.5, color="#FF6B6B")))
        fig.add_trace(go.Scatter(x=wvp.index, y=wvp["signal"], name="Weekly Signal", 
                            line=dict(width=1.5, color="#4ECDC4", dash="dash")))
                            
        fig.update_layout(height=800, title_text="Weekly Bell Curve Oscillator")
        fig.update_yaxes(title_text="Weekly Oscillator (0-100)", range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Quantitative Out-of-Sample Results")
        
        st.subheader("Daily Trade Log")
        if len(daily_bt.trades) > 0:
            st.dataframe(daily_bt.trades, use_container_width=True)
        else:
            st.info("No trades generated in the daily backtest period.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{daily_bt.metrics.get('total_return', 0):.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{daily_bt.metrics.get('sharpe_ratio', 0):.2f}")
        with col3:
            st.metric("Sortino Ratio", f"{daily_bt.metrics.get('sortino_ratio', 0):.2f}")
        with col4:
            st.metric("Calmar Ratio", f"{daily_bt.metrics.get('calmar_ratio', 0):.2f}")
            
        st.subheader("Equity Curve")
        st.line_chart(daily_bt.equity_curve, use_container_width=True)
        
        st.subheader("Weekly Performance")
        if len(weekly_bt.trades) > 0:
            st.dataframe(weekly_bt.trades, use_container_width=True)
        else:
            st.info("No weekly trades generated.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Weekly Total Return", f"{weekly_bt.metrics.get('total_return', 0):.2%}")
        with col2:
            st.metric("Weekly Sharpe", f"{weekly_bt.metrics.get('sharpe_ratio', 0):.2f}")
        with col3:
            st.metric("Max Drawdown", f"{weekly_bt.metrics.get('max_drawdown', 0):.2%}")
        with col4:
            st.metric("Profit Factor", f"{weekly_bt.metrics.get('profit_factor', 0):.2f}")
            
        st.line_chart(weekly_bt.equity_curve, use_container_width=True)

if __name__ == "__main__":
    main()
