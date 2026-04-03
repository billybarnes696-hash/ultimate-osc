"""
RSP HYBRID OSCILLATOR - STREAMLIT PRODUCTION v2.0
Features:
- Upload historical breadth ZIP (StockCharts format)
- Upload daily snapshot CSV for latest values
- Defeat Beta API integration
- Rate limit handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import zipfile
import io

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# ============================================================================

st.set_page_config(
    page_title="RSP Hybrid Oscillator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

def get_config():
    """Return configuration dict"""
    return {
        'lookback': 126,
        'sigma_smooth': 3.0,
        'price_weight': 0.30,
        'breadth_weight': 0.70,
        'breadth_weights': {
            'SPXA50R': 2.0,
            'BPSPX': 1.6,
            'BPNYA': 1.2,
            'NYMO': 1.3,
            'NYSI': 1.0,
            'NYAD': 0.9,
            'TRIN': 0.7,
            'CPCE': 0.7,
        },
        'bull_long_enter': 22.0,
        'bull_long_exit': 48.0,
        'bear_long_enter': 12.0,
        'bear_long_exit': 35.0,
        'sideways_long_enter': 18.0,
        'sideways_long_exit': 45.0,
        'target_vol': 0.12,
        'stop_loss_pct': 0.06,
        'min_quality': 0.45,
        'slippage_bps': 2.0,
        'spread_bps': 1.0,
        'ticker': 'RSP',
    }


# ============================================================================
# FILE PARSERS
# ============================================================================

def parse_stockcharts_pipe_csv(content: str) -> pd.Series:
    """Parse StockCharts pipe-delimited CSV format"""
    lines = content.strip().split('\n')
    dates = []
    values = []
    
    for line in lines:
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 5:
            try:
                # StockCharts format: Data|Date|Open|High|Low|Close|Volume
                # Or sometimes just Date|Close
                date_str = None
                close_str = None
                
                # Check if first line is header
                if parts[0].lower() == 'data' and len(parts) >= 6:
                    # Header row - find indices
                    for i, p in enumerate(parts):
                        if p.lower() == 'date':
                            date_idx = i
                        elif p.lower() == 'close':
                            close_idx = i
                    # Use next line for actual data (skip header)
                    continue
                else:
                    # Assume format: Data|Date|Open|High|Low|Close|Volume
                    date_idx = 1
                    close_idx = 5
                
                if len(parts) > max(date_idx, close_idx):
                    date_str = parts[date_idx]
                    close_str = parts[close_idx]
                
                if date_str and close_str and date_str.lower() != 'date':
                    date = pd.to_datetime(date_str)
                    close = float(close_str)
                    dates.append(date)
                    values.append(close)
            except Exception:
                continue
    
    if dates:
        series = pd.Series(values, index=dates).sort_index()
        series = series[~series.index.duplicated(keep='last')]
        return series
    return pd.Series(dtype=float)


def parse_daily_snapshot_csv(content: str) -> Dict[str, float]:
    """Parse daily snapshot CSV (Symbol, Close format)"""
    lines = content.strip().split('\n')
    snapshot = {}
    
    for line in lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            symbol = parts[0].strip().upper()
            try:
                # Handle $ symbols in symbol names
                if symbol.startswith('$'):
                    symbol = symbol
                value = float(parts[1])
                snapshot[symbol] = value
            except ValueError:
                continue
    
    return snapshot


# ============================================================================
# BREADTH LOADER WITH ZIP + SNAPSHOT
# ============================================================================

@st.cache_data(ttl=86400)
def load_breadth_from_zip(zip_file, snapshot_map=None, snap_date=None) -> Dict[str, pd.Series]:
    """Load breadth indicators from uploaded zip, apply snapshot for latest values"""
    
    if zip_file is None:
        return None
    
    breadth_data = {}
    
    with zipfile.ZipFile(io.BytesIO(zip_file.getvalue())) as zf:
        for filename in zf.namelist():
            if not filename.endswith('.csv'):
                continue
            
            stem = filename.replace('.csv', '').lower()
            
            # Match to known indicators
            matched = None
            for indicator in ['spxa50r', 'spxa200r', 'bpspx', 'bpnya', 'nymo', 'nysi', 'nyad', 'trin', 'cpc', 'cpce']:
                if indicator in stem:
                    matched = indicator.upper()
                    break
            
            if matched:
                content = zf.read(filename).decode('utf-8', errors='ignore')
                series = parse_stockcharts_pipe_csv(content)
                
                if not series.empty and len(series) > 50:
                    # Apply snapshot if available (for latest value)
                    if snapshot_map and snap_date:
                        # Map filename stem to snapshot symbol
                        snap_symbol_map = {
                            'SPXA50R': '$SPXA50R',
                            'SPXA200R': '$SPXA200R',
                            'BPSPX': '$BPSPX',
                            'BPNYA': '$BPNYA',
                            'NYMO': '$NYMO',
                            'NYSI': '$NYSI',
                            'NYAD': '$NYAD',
                            'TRIN': '$TRIN',
                            'CPC': '$CPCE',
                            'CPCE': '$CPCE',
                        }
                        snap_symbol = snap_symbol_map.get(matched)
                        if snap_symbol and snap_symbol in snapshot_map:
                            series.loc[snap_date] = snapshot_map[snap_symbol]
                            series = series.sort_index()
                            series = series[~series.index.duplicated(keep='last')]
                    
                    breadth_data[matched] = series
                    st.success(f"✓ Loaded {matched} ({len(series)} days)")
    
    return breadth_data if breadth_data else None


@st.cache_data(ttl=3600)
def load_breadth_from_yahoo() -> Dict[str, pd.Series]:
    """Fallback: load breadth from Yahoo Finance"""
    
    symbols = {
        'SPXA50R': '$SPXA50R',
        'BPSPX': '$BPSPX',
        'BPNYA': '$BPNYA',
        'NYMO': '$NYMO',
        'NYSI': '$NYSI',
        'NYAD': '$NYAD',
        'TRIN': '$TRIN',
        'CPCE': '$CPCE',
    }
    
    breadth_data = {}
    end_date = datetime.now()
    start_date = end_date.replace(year=end_date.year - 10)
    
    for name, symbol in symbols.items():
        for attempt in range(2):
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    if 'Adj Close' in df.columns:
                        series = df['Adj Close']
                    elif 'Close' in df.columns:
                        series = df['Close']
                    else:
                        series = df.iloc[:, 0]
                    
                    breadth_data[name] = series.dropna()
                    st.success(f"✓ Loaded {name} from Yahoo")
                break
            except Exception as e:
                if "RateLimit" in str(e) and attempt == 0:
                    time.sleep(2)
                    continue
                else:
                    st.warning(f"✗ Could not load {name}: {str(e)[:50]}")
                    break
    
    return breadth_data if breadth_data else None


# ============================================================================
# PRICE LOADER
# ============================================================================

@st.cache_data(ttl=3600)
def load_price_data(ticker: str, years: int) -> pd.Series:
    """Load price data from Yahoo Finance with rate limit handling"""
    
    end_date = datetime.now()
    start_date = end_date.replace(year=end_date.year - years)
    
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                return None
            
            if 'Adj Close' in df.columns:
                price = df['Adj Close']
            elif 'Close' in df.columns:
                price = df['Close']
            else:
                price = df.iloc[:, 0]
            
            return price.dropna()
            
        except Exception as e:
            if "RateLimit" in str(e) or "Too Many" in str(e):
                wait = (attempt + 1) * 2
                st.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                st.error(f"Error loading {ticker}: {e}")
                return None
    
    return None


# ============================================================================
# GAUSSIAN CDF TRANSFORM
# ============================================================================

def gaussian_cdf_transform(series: pd.Series, lookback: int, sigma: float) -> pd.Series:
    """Convert any series to 0-100 bell curve"""
    if series is None or len(series) < lookback:
        return pd.Series(dtype=float)
    
    s = series.copy()
    s = s.interpolate(limit_direction='both').ffill().bfill()
    
    rolling_mean = s.rolling(lookback, min_periods=max(20, lookback//3)).mean()
    rolling_std = s.rolling(lookback, min_periods=max(20, lookback//3)).std()
    
    z = (s - rolling_mean) / rolling_std.clip(lower=1e-8)
    z = z.clip(lower=-3.5, upper=3.5)
    
    cdf = norm.cdf(z) * 100.0
    
    if sigma > 0:
        smooth = gaussian_filter1d(cdf.to_numpy(dtype=float), sigma=sigma)
        out = pd.Series(smooth, index=series.index)
    else:
        out = pd.Series(cdf, index=series.index)
    
    out[rolling_mean.isna()] = np.nan
    return out


# ============================================================================
# REGIME DETECTION
# ============================================================================

def detect_regime(price: pd.Series) -> pd.Series:
    """Detect bull/bear/sideways market regime"""
    if price is None or len(price) < 252:
        return pd.Series('sideways', index=price.index if price is not None else [])
    
    rets = price.pct_change()
    trend_252 = price / price.shift(252) - 1.0
    trend_63 = price / price.shift(63) - 1.0
    
    vol_short = rets.rolling(20, min_periods=10).std() * np.sqrt(252)
    vol_long = rets.rolling(126, min_periods=60).std() * np.sqrt(252)
    vol_ratio = vol_short / vol_long.clip(lower=1e-8)
    
    dd = price / price.cummax() - 1.0
    
    regime = pd.Series("sideways", index=price.index, dtype=object)
    bull_mask = (trend_252 > 0.03) & (trend_63 > -0.02) & (vol_ratio < 1.20) & (dd > -0.10)
    bear_mask = (trend_252 < -0.03) | (trend_63 < -0.06) | (vol_ratio > 1.35) | (dd < -0.15)
    
    regime.loc[bull_mask] = "bull"
    regime.loc[bear_mask] = "bear"
    regime = regime.ffill().fillna("sideways")
    
    return regime


def get_thresholds(regime: str, config: dict):
    """Get regime-appropriate thresholds"""
    if regime == "bull":
        return config['bull_long_enter'], config['bull_long_exit']
    elif regime == "bear":
        return config['bear_long_enter'], config['bear_long_exit']
    else:
        return config['sideways_long_enter'], config['sideways_long_exit']


# ============================================================================
# ULTIMATE OSCILLATOR
# ============================================================================

def calculate_ultimate_oscillator(price: pd.Series, breadth_data: dict, config: dict) -> pd.DataFrame:
    """Calculate the ultimate oscillator with weighted breadth"""
    
    # Price oscillator
    price_osc = gaussian_cdf_transform(price, config['lookback'], config['sigma_smooth'])
    
    if not breadth_data:
        # Price-only fallback
        df = pd.DataFrame({
            'price': price,
            'oscillator': price_osc,
            'quality': 0.5,
        })
        return df
    
    # Transform each breadth component
    breadth_curves = {}
    for name, series in breadth_data.items():
        if name in config['breadth_weights']:
            # Apply polarity
            if name in ['TRIN', 'CPCE']:
                series = -series
            
            # Special handling for NYAD
            if name == 'NYAD':
                series = series.pct_change(5)
            
            breadth_curves[name] = gaussian_cdf_transform(
                series, config['lookback'], config['sigma_smooth']
            )
    
    if not breadth_curves:
        df = pd.DataFrame({
            'price': price,
            'oscillator': price_osc,
            'quality': 0.5,
        })
        return df
    
    # Weighted composite
    breadth_df = pd.DataFrame(breadth_curves).dropna(how='all')
    weights = pd.Series(config['breadth_weights'])
    common_cols = [c for c in breadth_df.columns if c in weights.index]
    
    weighted_sum = 0
    weight_total = 0
    for col in common_cols:
        weighted_sum += breadth_df[col] * weights[col]
        weight_total += weights[col]
    
    breadth_curve = weighted_sum / weight_total
    
    # Quality score
    quality = pd.Series(0.5, index=price.index)
    if 'SPXA50R' in breadth_curves:
        quality = (breadth_curves['SPXA50R'] > 30).astype(float) * 0.5
        quality += (breadth_curves['SPXA50R'] > 40).astype(float) * 0.3
    
    # Final oscillator
    final_osc = config['breadth_weight'] * breadth_curve + config['price_weight'] * price_osc
    
    df = pd.DataFrame({
        'price': price,
        'oscillator': final_osc,
        'quality': quality.clip(0, 1),
        'breadth_curve': breadth_curve,
        'price_curve': price_osc,
    })
    
    return df


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(df: pd.DataFrame, config: dict) -> dict:
    """Run backtest with hysteresis and position sizing"""
    
    if df is None or len(df) < 100:
        return {'error': 'Insufficient data'}
    
    work = df.dropna(subset=['price', 'oscillator']).copy()
    if len(work) < 100:
        return {'error': 'Insufficient data after cleaning'}
    
    returns = work['price'].pct_change().fillna(0)
    regime = detect_regime(work['price'])
    
    # Volatility scaling
    vol = returns.rolling(20, min_periods=10).std() * np.sqrt(252)
    vol_scale = (config['target_vol'] / vol.clip(lower=0.05, upper=0.30)).clip(0.25, 1.0).fillna(0.5)
    
    position = pd.Series(0.0, index=work.index)
    strategy_returns = pd.Series(0.0, index=work.index)
    trades = []
    
    in_position = False
    entry_price = 0.0
    entry_date = None
    bars_held = 0
    max_bars = 20
    
    for i in range(1, len(work)):
        date = work.index[i]
        curr_osc = float(work['oscillator'].iloc[i])
        curr_quality = float(work['quality'].iloc[i])
        curr_price = float(work['price'].iloc[i])
        curr_regime = regime.iloc[i]
        
        long_enter, long_exit = get_thresholds(curr_regime, config)
        
        # Entry condition
        entry_signal = (curr_osc <= long_enter) and (curr_quality >= config['min_quality'])
        
        # Exit condition
        exit_signal = (curr_osc >= long_exit) or (curr_quality < 0.20) or (bars_held >= max_bars)
        
        # Stop loss
        stop_hit = False
        if in_position:
            pnl_pct = (curr_price - entry_price) / entry_price
            if pnl_pct <= -config['stop_loss_pct']:
                stop_hit = True
        
        if not in_position and entry_signal:
            in_position = True
            entry_price = curr_price
            entry_date = date
            bars_held = 0
            
            # Transaction cost
            cost_bps = config['slippage_bps'] + config['spread_bps']
            strategy_returns.iloc[i] -= cost_bps / 10000
            
        elif in_position and (exit_signal or stop_hit):
            exit_price = curr_price
            pnl = (exit_price / entry_price) - 1
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'pnl': round(pnl, 4),
                'bars_held': bars_held,
                'exit_reason': 'stop_loss' if stop_hit else 'signal',
                'regime': curr_regime
            })
            
            in_position = False
            
            # Transaction cost on exit
            cost_bps = config['slippage_bps'] + config['spread_bps']
            strategy_returns.iloc[i] -= cost_bps / 10000
        
        position.iloc[i] = 1.0 if in_position else 0.0
        strategy_returns.iloc[i] += returns.iloc[i] * position.iloc[i] * vol_scale.iloc[i]
        
        if in_position:
            bars_held += 1
    
    # Close open position at end
    if in_position and len(trades) > 0:
        trades[-1]['exit_date'] = work.index[-1]
        trades[-1]['exit_price'] = float(work['price'].iloc[-1])
        trades[-1]['pnl'] = (trades[-1]['exit_price'] / trades[-1]['entry_price']) - 1
    
    # Equity curve
    equity = (1 + strategy_returns.fillna(0)).cumprod()
    
    # Calculate metrics
    n = len(strategy_returns.dropna())
    years = n / 252
    
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    volatility = float(strategy_returns.std() * np.sqrt(252))
    sharpe = cagr / volatility if volatility > 0 else 0
    
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())
    
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        win_rate = float((trades_df['pnl'] > 0).mean())
        num_trades = len(trades_df)
    else:
        win_rate = 0
        num_trades = 0
    
    # Buy & hold comparison
    bh_return = (work['price'].iloc[-1] / work['price'].iloc[0] - 1)
    bh_cagr = (1 + bh_return) ** (1 / years) - 1 if years > 0 else 0
    outperformance = cagr - bh_cagr
    
    return {
        'metrics': {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'bh_cagr': bh_cagr,
            'outperformance': outperformance,
        },
        'equity_curve': equity,
        'position': position,
        'trades': trades_df,
        'oscillator': work['oscillator'],
        'quality': work['quality'],
    }


# ============================================================================
# PLOTTING
# ============================================================================

def create_plots(results: dict):
    """Create interactive Plotly charts"""
    
    equity = results.get('equity_curve')
    position = results.get('position')
    oscillator = results.get('oscillator')
    quality = results.get('quality')
    
    if equity is None or equity.empty:
        return None
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.35, 0.25, 0.20, 0.20],
        subplot_titles=("Equity Curve", "Oscillator (0-100)", "Quality Score", "Position (1=Long, 0=Cash)")
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity, name="Strategy", line=dict(color='#2ECC71', width=2)),
        row=1, col=1
    )
    
    # Oscillator
    if oscillator is not None:
        fig.add_trace(
            go.Scatter(x=oscillator.index, y=oscillator, name="Oscillator", line=dict(color='#E74C3C', width=1.5)),
            row=2, col=1
        )
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1,
                      annotation_text="Buy Zone", annotation_position="right")
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1,
                      annotation_text="Cash Zone", annotation_position="right")
    
    # Quality
    if quality is not None:
        fig.add_trace(
            go.Scatter(x=quality.index, y=quality, name="Quality", fill='tozeroy', 
                      line=dict(color='#F39C12', width=1)),
            row=3, col=1
        )
        fig.add_hline(y=0.45, line_dash="dash", line_color="blue", row=3, col=1)
    
    # Position
    fig.add_trace(
        go.Scatter(x=position.index, y=position, name="Position", fill='tozeroy',
                  line=dict(color='#9B59B6', width=1)),
        row=4, col=1
    )
    
    fig.update_layout(height=900, title_text="RSP Hybrid Oscillator Strategy", showlegend=True)
    fig.update_yaxes(title_text="Equity (starting at 1)", row=1, col=1)
    fig.update_yaxes(title_text="Oscillator", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Quality", range=[0, 1], row=3, col=1)
    fig.update_yaxes(title_text="Position", range=[0, 1], row=4, col=1)
    
    return fig


# ============================================================================
# MAIN STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("📊 RSP Hybrid Oscillator")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        years = st.slider("Historical Years", 5, 25, 20)
        
        st.markdown("---")
        st.header("📁 Data Upload")
        
        # Breadth ZIP upload
        st.subheader("Historical Breadth Data")
        breadth_zip = st.file_uploader(
            "Upload StockCharts Breadth ZIP",
            type=['zip'],
            help="Upload the zip file containing breadth indicator CSVs from StockCharts"
        )
        
        # Daily snapshot CSV upload
        st.subheader("Daily Snapshot")
        snapshot_file = st.file_uploader(
            "Upload Daily Snapshot CSV",
            type=['csv'],
            help="CSV with columns: Symbol, Close (e.g., $SPXA50R, 45.2)"
        )
        
        if snapshot_file:
            st.success(f"✅ Loaded: {snapshot_file.name}")
        
        st.markdown("---")
        st.header("📈 Thresholds")
        
        col1, col2 = st.columns(2)
        with col1:
            bull_enter = st.number_input("Bull Long Enter", value=22.0, step=1.0)
        with col2:
            bull_exit = st.number_input("Bull Long Exit", value=48.0, step=1.0)
        
        st.markdown("---")
        st.header("⚠️ Risk")
        
        stop_loss = st.slider("Stop Loss %", 2.0, 10.0, 6.0, step=0.5)
        target_vol = st.slider("Target Volatility %", 8.0, 20.0, 12.0, step=1.0)
        
        st.markdown("---")
        st.caption("Data Sources: Historical ZIP | Yahoo Finance for latest")
    
    # Update config with user inputs
    config = get_config()
    config['bull_long_enter'] = bull_enter
    config['bull_long_exit'] = bull_exit
    config['stop_loss_pct'] = stop_loss / 100
    config['target_vol'] = target_vol / 100
    
    # Parse snapshot if uploaded
    snapshot_map = None
    snap_date = None
    
    if snapshot_file:
        try:
            content = snapshot_file.getvalue().decode('utf-8')
            snapshot_map = parse_daily_snapshot_csv(content)
            snap_date = pd.Timestamp.now().normalize()
            st.success(f"✅ Parsed {len(snapshot_map)} symbols from snapshot")
        except Exception as e:
            st.error(f"Error parsing snapshot: {e}")
    
    # Load data button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        
        # Load price data
        with st.spinner("Loading RSP price data..."):
            price = load_price_data(config['ticker'], years)
        
        if price is None:
            st.error("Failed to load price data. Please try again.")
            return
        
        st.success(f"✅ Loaded {len(price)} days of price data")
        
        # Load breadth data (prioritize uploaded zip)
        breadth = None
        
        if breadth_zip:
            with st.spinner("Loading breadth from uploaded zip..."):
                breadth = load_breadth_from_zip(breadth_zip, snapshot_map, snap_date)
        
        if not breadth:
            with st.spinner("Loading breadth from Yahoo Finance..."):
                breadth = load_breadth_from_yahoo()
        
        if breadth:
            st.success(f"✅ Loaded {len(breadth)} breadth indicators")
        else:
            st.warning("⚠️ No breadth data available. Using price-only oscillator.")
        
        # Calculate oscillator
        with st.spinner("Calculating oscillator..."):
            df = calculate_ultimate_oscillator(price, breadth, config)
        
        # Run backtest
        with st.spinner("Running backtest..."):
            results = run_backtest(df, config)
        
        if 'error' in results:
            st.error(results['error'])
            return
        
        # Display metrics
        metrics = results['metrics']
        
        st.markdown("---")
        st.header("📊 Performance Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{metrics['total_return']:.2%}")
            st.metric("CAGR", f"{metrics['cagr']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
            st.metric("Volatility", f"{metrics['volatility']:.2%}")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
        with col4:
            st.metric("Buy & Hold CAGR", f"{metrics['bh_cagr']:.2%}")
            st.metric("Outperformance", f"{metrics['outperformance']:.2%}", 
                     delta_color="normal" if metrics['outperformance'] > 0 else "inverse")
        
        st.markdown("---")
        st.header("📈 Charts")
        
        # Plot results
        fig = create_plots(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade log
        trades = results.get('trades')
        if trades is not None and not trades.empty:
            st.markdown("---")
            st.header("📋 Recent Trades")
            
            display_trades = trades.tail(20).copy()
            display_trades['pnl_pct'] = display_trades['pnl'] * 100
            display_trades['entry_date'] = pd.to_datetime(display_trades['entry_date']).dt.date
            display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date']).dt.date
            
            st.dataframe(
                display_trades[['entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl_pct', 'bars_held', 'exit_reason', 'regime']],
                use_container_width=True,
                hide_index=True
            )
        
        # Current signal
        st.markdown("---")
        st.header("🔔 Current Signal")
        
        last_osc = results['oscillator'].iloc[-1]
        last_quality = results['quality'].iloc[-1]
        last_price = df['price'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Oscillator", f"{last_osc:.1f}")
        with col2:
            st.metric("Quality Score", f"{last_quality:.2f}")
        with col3:
            st.metric("Current Price", f"${last_price:.2f}")
        
        # Determine signal
        if last_osc <= config['bull_long_enter'] and last_quality >= config['min_quality']:
            st.success(f"🔴 **BUY SIGNAL** - Oscillator ({last_osc:.1f}) is oversold and quality ({last_quality:.2f}) is good")
        elif last_osc >= config['bull_long_exit']:
            st.info(f"🟢 **CASH SIGNAL** - Oscillator ({last_osc:.1f}) is overbought, take profits")
        else:
            st.warning(f"⚪ **NEUTRAL** - Wait for oversold condition (oscillator < {config['bull_long_enter']:.0f})")
    
    else:
        # Welcome screen before button click
        st.info("👈 Configure settings in the sidebar, upload breadth ZIP/snapshot, then click **Run Backtest**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📖 How It Works
            
            **Oscillator (0-100 scale):**
            - **< 20** → Oversold (BUY zone)
            - **20-80** → Neutral (CASH)
            - **> 80** → Overbought (CASH)
            
            **Strategy Rules:**
            - Buy RSP when oscillator < 20 AND quality > 0.45
            - Hold cash otherwise
            - 6% stop loss protection
            """)
        
        with col2:
            st.markdown("""
            ### 📁 Data Upload Options
            
            **1. Historical Breadth ZIP**
            - StockCharts export format
            - Contains historical data for backtesting
            - Files like _spxa50r.csv, _Bpspx.csv
            
            **2. Daily Snapshot CSV**
            - Format: Symbol, Close
            - Example: `$SPXA50R, 45.2`
            - Updates latest values only
            """)
        
        st.markdown("---")
        
        # Show expected file formats
        with st.expander("📄 Expected File Formats"):
            st.markdown("""
            **Breadth ZIP Contents:**
