
import io
import zipfile
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

warnings.filterwarnings("ignore")

st.set_page_config(page_title="RSP Ultimate Oscillator", layout="wide")


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class AppConfig:
    lookback: int = 126
    smooth_price: float = 4.0
    smooth_breadth: float = 4.0
    breadth_weight: float = 0.70
    price_weight: float = 0.30
    target_vol: float = 0.15
    stop_loss_pct: float = 0.06
    min_quality_for_long: float = 0.45
    breadth_weights: Dict[str, float] = field(default_factory=lambda: {
        "_spxa50r": 2.2,
        "_spxa200r": 1.8,
        "_Bpspx": 1.6,
        "_Bpnya": 1.2,
        "_nymo": 1.3,
        "_nySI": 1.0,
        "_nyad": 0.9,
        "_trin": 0.7,
        "_cpc": 0.7,
    })


# ============================================================================
# HELPERS
# ============================================================================

def normalize_price_series_from_yf(df: pd.DataFrame, symbol: str) -> pd.Series:
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", symbol) in df.columns:
            s = df[("Close", symbol)]
        elif ("Adj Close", symbol) in df.columns:
            s = df[("Adj Close", symbol)]
        else:
            first_close = [c for c in df.columns if c[0] in ("Close", "Adj Close")]
            if not first_close:
                raise ValueError(f"Could not find Close/Adj Close for {symbol}")
            s = df[first_close[0]]
    else:
        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            raise ValueError(f"Could not find Close/Adj Close for {symbol}")

    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = "Close"
    return s


def bars_from_window(label: str) -> int:
    mapping = {
        "3M": 63,
        "6M": 126,
        "1Y": 252,
        "2Y": 504,
        "5Y": 1260,
        "10Y": 2520,
        "Max": 0,
    }
    return mapping.get(label, 252)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(show_spinner=False)
def fetch_price_history(symbol: str, period: str) -> pd.Series:
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    return normalize_price_series_from_yf(df, symbol)


@st.cache_data(show_spinner=False)
def parse_price_csv(file_bytes: bytes, filename: str) -> pd.Series:
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols = {c.lower().strip(): c for c in df.columns}

    date_col = None
    price_col = None

    for c in ["date", "datetime", "timestamp"]:
        if c in cols:
            date_col = cols[c]
            break

    for c in ["close", "adj close", "adj_close", "price"]:
        if c in cols:
            price_col = cols[c]
            break

    if date_col is None or price_col is None:
        raise ValueError(f"{filename}: need a date column and a close/price column.")

    out = df[[date_col, price_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna().sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    s = out.set_index(date_col)[price_col].astype(float)
    s.name = "Close"
    return s


# ============================================================================
# CORE QUANT ENGINE
# ============================================================================

class BellCurveTransform:
    @staticmethod
    def calculate(
        series: pd.Series,
        lookback: int = 126,
        sigma: float = 4.0,
        z_clip: float = 3.5,
        min_frac: float = 0.67,
    ) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").copy()
        s = s.interpolate(limit_direction="both").ffill().bfill()

        min_periods = max(20, int(lookback * min_frac))
        rolling_mean = s.rolling(lookback, min_periods=min_periods).mean()
        rolling_std = s.rolling(lookback, min_periods=min_periods).std().clip(lower=1e-8)

        z = ((s - rolling_mean) / rolling_std).clip(-z_clip, z_clip)
        cdf = norm.cdf(z) * 100.0

        if sigma > 0:
            smooth = gaussian_filter1d(cdf.to_numpy(dtype=float), sigma=sigma)
            out = pd.Series(smooth, index=series.index)
        else:
            out = pd.Series(cdf, index=series.index)

        out[rolling_mean.isna()] = np.nan
        return out


class BreadthThrustDetector:
    @staticmethod
    def calculate(
        breadth_osc: pd.Series,
        spxa50r_raw: Optional[pd.Series] = None,
        bpspx_raw: Optional[pd.Series] = None,
    ) -> pd.Series:
        bo = pd.to_numeric(breadth_osc, errors="coerce")
        impulse_5 = bo.diff(5)
        impulse_10 = bo.diff(10)

        thrust = pd.Series(0.0, index=bo.index)
        thrust += (impulse_5 > 12).astype(float) * 0.40
        thrust += (impulse_10 > 18).astype(float) * 0.35
        thrust += ((bo > 35) & (bo.shift(3) < 20)).astype(float) * 0.25

        if spxa50r_raw is not None:
            s50 = pd.to_numeric(spxa50r_raw, errors="coerce")
            thrust += ((s50 > 30) & (s50.diff(3) > 2)).astype(float) * 0.25

        if bpspx_raw is not None:
            bp = pd.to_numeric(bpspx_raw, errors="coerce")
            thrust += ((bp.diff(5) > 1.5) | (bp > bp.rolling(20, min_periods=10).mean())).astype(float) * 0.15

        return thrust.clip(lower=0.0, upper=1.0)


class RegimeDetector:
    def detect_regime(self, price: pd.Series) -> pd.Series:
        px = pd.to_numeric(price, errors="coerce")
        rets = px.pct_change()

        trend_252 = px / px.shift(252) - 1.0
        trend_63 = px / px.shift(63) - 1.0
        vol_short = rets.rolling(20, min_periods=10).std() * np.sqrt(252)
        vol_long = rets.rolling(126, min_periods=60).std() * np.sqrt(252)
        vol_ratio = vol_short / vol_long.clip(lower=1e-8)
        drawdown = px / px.cummax() - 1.0

        regime = pd.Series("sideways", index=px.index, dtype=object)
        bull_mask = (
            (trend_252 > 0.03)
            & (trend_63 > -0.02)
            & (vol_ratio < 1.20)
            & (drawdown > -0.10)
        )
        bear_mask = (
            (trend_252 < -0.03)
            | (trend_63 < -0.06)
            | (vol_ratio > 1.35)
            | (drawdown < -0.15)
        )

        regime.loc[bull_mask] = "bull"
        regime.loc[bear_mask] = "bear"
        return regime.ffill().fillna("sideways")

    def get_thresholds(self, regime: str) -> Dict[str, float]:
        if regime == "bull":
            return {"long_enter": 22.0, "long_exit": 48.0, "short_enter": 92.0, "short_exit": 60.0}
        if regime == "bear":
            return {"long_enter": 12.0, "long_exit": 35.0, "short_enter": 72.0, "short_exit": 50.0}
        return {"long_enter": 18.0, "long_exit": 45.0, "short_enter": 82.0, "short_exit": 55.0}


class UltimateOscillator:
    def __init__(self, config: AppConfig):
        self.config = config

    def _parse_stockcharts_pipe_text(self, text: str) -> pd.Series:
        lines = text.strip().split("\n")
        rows = []
        for line in lines:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 5:
                try:
                    dt = pd.to_datetime(parts[1], errors="coerce")
                    close = float(parts[4])
                    if pd.notna(dt):
                        rows.append((dt, close))
                except Exception:
                    pass

        if not rows:
            return pd.Series(dtype=float)

        s = pd.Series({dt: close for dt, close in rows}).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s.astype(float)

    def load_breadth_from_zip(self, zip_bytes: bytes) -> Dict[str, pd.Series]:
        components = {}
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            for stem in self.config.breadth_weights:
                matches = [n for n in names if stem.lower() in n.lower() and n.lower().endswith(".csv")]
                if not matches:
                    continue
                text = zf.read(matches[0]).decode("utf-8", errors="ignore")
                s = self._parse_stockcharts_pipe_text(text)
                if not s.empty:
                    components[stem] = s
        return components

    def transform_component(self, stem: str, series: pd.Series) -> pd.Series:
        x = pd.to_numeric(series, errors="coerce").copy()

        if stem == "_nyad":
            x = x.pct_change(5)
        elif stem in ("_trin", "_cpc"):
            x = -x

        return BellCurveTransform.calculate(
            x,
            lookback=self.config.lookback,
            sigma=self.config.smooth_breadth,
        )

    def calculate(self, price: pd.Series, breadth_components: Optional[Dict[str, pd.Series]]) -> pd.DataFrame:
        price_curve = BellCurveTransform.calculate(
            price,
            lookback=self.config.lookback,
            sigma=self.config.smooth_price,
        )

        if not breadth_components:
            df = pd.DataFrame({
                "price": price,
                "price_curve": price_curve,
                "breadth_curve": price_curve,
                "final_osc": price_curve,
                "quality_score": 0.5,
                "thrust_score": 0.0,
            })
            return df.dropna(subset=["price"])

        transformed = {}
        raw = {}
        for stem, series in breadth_components.items():
            raw[stem] = pd.to_numeric(series, errors="coerce")
            transformed[stem] = self.transform_component(stem, series)

        breadth_df = pd.DataFrame(transformed).sort_index()
        weights = pd.Series(self.config.breadth_weights, dtype=float)
        cols = [c for c in breadth_df.columns if c in weights.index]

        if not cols:
            df = pd.DataFrame({
                "price": price,
                "price_curve": price_curve,
                "breadth_curve": price_curve,
                "final_osc": price_curve,
                "quality_score": 0.5,
                "thrust_score": 0.0,
            })
            return df.dropna(subset=["price"])

        w = weights.loc[cols]
        weighted_sum = breadth_df[cols].multiply(w, axis=1).sum(axis=1, min_count=1)
        weight_sum = breadth_df[cols].notna().multiply(w, axis=1).sum(axis=1).replace(0, np.nan)
        breadth_curve = weighted_sum / weight_sum

        thrust = BreadthThrustDetector.calculate(
            breadth_curve,
            spxa50r_raw=raw.get("_spxa50r"),
            bpspx_raw=raw.get("_Bpspx"),
        )

        quality = pd.Series(0.0, index=breadth_curve.index)
        if "_spxa50r" in raw:
            s50 = raw["_spxa50r"].reindex(quality.index)
            quality += (s50 > 30).astype(float) * 0.40
            quality += (s50 > 40).astype(float) * 0.20
        if "_Bpspx" in raw:
            bp = raw["_Bpspx"].reindex(quality.index)
            quality += (bp > 20).astype(float) * 0.20
        quality += thrust.fillna(0.0) * 0.20
        quality = quality.clip(0.0, 1.0)

        df = pd.DataFrame({
            "price": price,
            "price_curve": price_curve,
            "breadth_curve": breadth_curve,
            "final_osc": (
                self.config.breadth_weight * breadth_curve
                + self.config.price_weight * price_curve
            ),
            "quality_score": quality,
            "thrust_score": thrust,
        })

        for col in cols:
            df[col] = breadth_df[col]

        return df.dropna(subset=["price"])


class BacktestEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.regime_detector = RegimeDetector()

    def transaction_cost_decimal(self, position_size: float) -> float:
        slippage_bps = 2.0
        market_impact_bps = 5.0
        spread_bps = 1.0
        commission = 0.0005
        total_bps = (
            slippage_bps
            + market_impact_bps * np.sqrt(max(position_size, 1e-9))
            + spread_bps
            + commission * 10000 / max(position_size, 1e-9)
        )
        return total_bps / 10000.0

    def volatility_scaled_size(self, returns: pd.Series) -> pd.Series:
        vol = returns.rolling(20, min_periods=10).std() * np.sqrt(252)
        size = self.config.target_vol / vol.clip(lower=0.05, upper=0.30)
        return size.clip(lower=0.25, upper=1.00).fillna(0.50)

    def run(self, df: pd.DataFrame) -> Dict[str, object]:
        work = df.copy().dropna(subset=["price", "final_osc"])
        if len(work) < 100:
            return {
                "metrics": {},
                "equity_curve": pd.Series(dtype=float),
                "position": pd.Series(dtype=float),
                "trades": pd.DataFrame(),
                "returns": pd.Series(dtype=float),
            }

        returns = work["price"].pct_change().fillna(0.0)
        vol_scale = self.volatility_scaled_size(returns)
        regime = self.regime_detector.detect_regime(work["price"])

        position = pd.Series(0.0, index=work.index)
        strategy_returns = pd.Series(0.0, index=work.index)
        trades = []

        in_position = False
        entry_price = np.nan
        entry_date = None
        entry_osc = np.nan
        bars_held = 0
        max_bars = 20

        for i in range(1, len(work)):
            dt = work.index[i]
            curr_price = float(work["price"].iloc[i])
            curr_osc = float(work["final_osc"].iloc[i])
            curr_quality = float(work["quality_score"].iloc[i]) if "quality_score" in work else 0.0
            curr_thrust = float(work["thrust_score"].iloc[i]) if "thrust_score" in work else 0.0
            curr_regime = str(regime.iloc[i])
            th = self.regime_detector.get_thresholds(curr_regime)

            entry_signal = (
                curr_osc <= th["long_enter"]
                and (curr_quality >= self.config.min_quality_for_long or curr_thrust >= 0.60)
            )

            exit_signal = (
                curr_osc >= th["long_exit"]
                or (curr_quality < 0.20 and curr_thrust < 0.20)
                or bars_held >= max_bars
            )

            stop_hit = False
            if in_position and pd.notna(entry_price):
                pnl_pct = (curr_price - entry_price) / entry_price
                if pnl_pct <= -self.config.stop_loss_pct:
                    stop_hit = True

            if not in_position and entry_signal:
                in_position = True
                entry_price = curr_price
                entry_date = dt
                entry_osc = curr_osc
                bars_held = 0
                strategy_returns.iloc[i] -= self.transaction_cost_decimal(float(vol_scale.iloc[i]))

            elif in_position and (exit_signal or stop_hit):
                exit_price = curr_price
                pnl = exit_price / entry_price - 1.0
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": dt,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "bars_held": bars_held,
                    "entry_osc": entry_osc,
                    "exit_osc": curr_osc,
                    "exit_reason": "stop_loss" if stop_hit else "signal",
                    "regime": curr_regime,
                })
                in_position = False
                entry_price = np.nan
                entry_date = None
                entry_osc = np.nan
                bars_held = 0
                strategy_returns.iloc[i] -= self.transaction_cost_decimal(float(vol_scale.iloc[i]))

            position.iloc[i] = 1.0 if in_position else 0.0
            strategy_returns.iloc[i] += returns.iloc[i] * position.iloc[i] * float(vol_scale.iloc[i])

            if in_position:
                bars_held += 1

        if in_position and entry_date is not None and pd.notna(entry_price):
            exit_price = float(work["price"].iloc[-1])
            exit_osc = float(work["final_osc"].iloc[-1])
            pnl = exit_price / entry_price - 1.0
            trades.append({
                "entry_date": entry_date,
                "exit_date": work.index[-1],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "bars_held": bars_held,
                "entry_osc": entry_osc,
                "exit_osc": exit_osc,
                "exit_reason": "end_of_data",
                "regime": str(regime.iloc[-1]),
            })

        equity = (1.0 + strategy_returns.fillna(0.0)).cumprod()
        trades_df = pd.DataFrame(trades)
        metrics = self._metrics(equity, strategy_returns, trades_df)

        return {
            "metrics": metrics,
            "equity_curve": equity,
            "position": position,
            "trades": trades_df,
            "returns": strategy_returns,
        }

    def _metrics(self, equity: pd.Series, returns: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
        if equity.empty or len(equity) < 2:
            return {}

        n = len(returns.dropna())
        years = max(n / 252.0, 1 / 252.0)

        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else np.nan
        volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else np.nan
        sharpe = float(cagr / volatility) if volatility and volatility > 0 and pd.notna(cagr) else np.nan

        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        max_dd = float(drawdown.min()) if not drawdown.empty else np.nan

        if not trades.empty:
            wins = trades["pnl"] > 0
            losses = trades["pnl"] < 0
            win_rate = float(wins.mean())
            avg_win = float(trades.loc[wins, "pnl"].mean()) if wins.any() else 0.0
            avg_loss = float(trades.loc[losses, "pnl"].mean()) if losses.any() else 0.0
            profit_factor = float(abs(avg_win * win_rate / (avg_loss * (1 - win_rate)))) if avg_loss != 0 else np.inf
            num_trades = int(len(trades))
        else:
            win_rate = np.nan
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = np.nan
            num_trades = 0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "num_trades": num_trades,
        }


def summarize_signal(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {}

    regime_detector = RegimeDetector()
    regime = regime_detector.detect_regime(df["price"])
    last_idx = df.index[-1]
    last_regime = str(regime.loc[last_idx])
    th = regime_detector.get_thresholds(last_regime)

    osc = float(df["final_osc"].iloc[-1])
    quality = float(df["quality_score"].iloc[-1]) if "quality_score" in df else 0.0
    thrust = float(df["thrust_score"].iloc[-1]) if "thrust_score" in df else 0.0

    if osc <= th["long_enter"] and (quality >= 0.45 or thrust >= 0.60):
        state = "LONG"
    elif osc >= th["short_enter"]:
        state = "RISK-OFF / TAKE PROFITS"
    else:
        state = "HOLD / NEUTRAL"

    return {
        "date": last_idx,
        "regime": last_regime,
        "state": state,
        "final_osc": osc,
        "quality_score": quality,
        "thrust_score": thrust,
        "long_enter_threshold": th["long_enter"],
        "long_exit_threshold": th["long_exit"],
        "short_enter_threshold": th["short_enter"],
        "short_exit_threshold": th["short_exit"],
    }


# ============================================================================
# CACHED MODEL BUILD
# ============================================================================

@st.cache_data(show_spinner=False)
def build_model(
    price_mode: str,
    symbol: str,
    period: str,
    uploaded_price_bytes: Optional[bytes],
    uploaded_price_name: Optional[str],
    uploaded_zip_bytes: Optional[bytes],
    lookback: int,
    smooth_price: float,
    smooth_breadth: float,
    breadth_weight: float,
    stop_loss_pct: float,
    target_vol: float,
):
    if price_mode == "Upload CSV":
        if uploaded_price_bytes is None:
            raise ValueError("Upload CSV price source selected, but no price CSV was uploaded.")
        price = parse_price_csv(uploaded_price_bytes, uploaded_price_name or "price.csv")
    else:
        price = fetch_price_history(symbol, period)

    config = AppConfig(
        lookback=lookback,
        smooth_price=smooth_price,
        smooth_breadth=smooth_breadth,
        breadth_weight=breadth_weight,
        price_weight=1.0 - breadth_weight,
        stop_loss_pct=stop_loss_pct,
        target_vol=target_vol,
    )

    engine = UltimateOscillator(config)

    breadth_components = None
    if uploaded_zip_bytes is not None:
        breadth_components = engine.load_breadth_from_zip(uploaded_zip_bytes)

    osc_df = engine.calculate(price, breadth_components).sort_index()
    bt = BacktestEngine(config).run(osc_df)
    summary = summarize_signal(osc_df)

    return osc_df, bt, summary


# ============================================================================
# CHARTS
# ============================================================================

def trim_df(df: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
    if lookback_bars == 0 or len(df) <= lookback_bars:
        return df.copy()
    return df.iloc[-lookback_bars:].copy()


def make_oscillator_chart(df: pd.DataFrame, view_mode: str, lookback_bars: int) -> go.Figure:
    plot_df = trim_df(df, lookback_bars)
    fig = go.Figure()

    if view_mode in ("Combined", "All"):
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["final_osc"], mode="lines", name="Final Oscillator", line=dict(width=3)))

    if view_mode in ("Breadth", "All") and "breadth_curve" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["breadth_curve"], mode="lines", name="Breadth Curve", line=dict(width=2, dash="dot")))

    if view_mode in ("Price", "All") and "price_curve" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["price_curve"], mode="lines", name="Price Curve", line=dict(width=2, dash="dash")))

    fig.add_hrect(y0=0, y1=20, opacity=0.12, line_width=0)
    fig.add_hrect(y0=20, y1=40, opacity=0.08, line_width=0)
    fig.add_hrect(y0=40, y1=60, opacity=0.05, line_width=0)
    fig.add_hrect(y0=60, y1=80, opacity=0.08, line_width=0)
    fig.add_hrect(y0=80, y1=100, opacity=0.12, line_width=0)

    if not plot_df.empty and pd.notna(plot_df["final_osc"].iloc[-1]):
        fig.add_trace(go.Scatter(
            x=[plot_df.index[-1]],
            y=[plot_df["final_osc"].iloc[-1]],
            mode="markers",
            name="Current",
            marker=dict(size=12, symbol="diamond"),
        ))

    fig.update_layout(
        title="Ultimate Oscillator",
        height=520,
        yaxis_title="0–100 Score",
        xaxis_title="Date",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_price_chart(df: pd.DataFrame, lookback_bars: int) -> go.Figure:
    plot_df = trim_df(df, lookback_bars)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["price"], mode="lines", name="Price", line=dict(width=2)))
    fig.update_layout(
        title="Price",
        height=350,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_equity_chart(backtest: Dict[str, object]) -> go.Figure:
    fig = go.Figure()
    eq = backtest.get("equity_curve", pd.Series(dtype=float))
    if isinstance(eq, pd.Series) and not eq.empty:
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Strategy Equity", line=dict(width=2)))
    fig.update_layout(
        title="Backtest Equity Curve",
        height=350,
        xaxis_title="Date",
        yaxis_title="Equity",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ============================================================================
# UI
# ============================================================================

st.title("RSP Ultimate Oscillator")
st.caption("Single-file Streamlit build designed to reduce deployment and import errors.")

with st.sidebar:
    st.header("Inputs")

    price_mode = st.radio("Price source", ["Auto-download", "Upload CSV"], index=0)
    symbol = st.text_input("Ticker", value="RSP")
    period = st.selectbox("Auto-download history", ["2y", "5y", "10y", "max"], index=2)

    price_file = None
    if price_mode == "Upload CSV":
        price_file = st.file_uploader(
            "Upload price CSV",
            type=["csv"],
            help="Needs a date column and a close/price column.",
        )

    breadth_zip = st.file_uploader(
        "Upload breadth ZIP",
        type=["zip"],
        help="Optional StockCharts breadth ZIP.",
    )

    st.header("Model")
    lookback = st.slider("Bell curve lookback", 63, 252, 126, step=21)
    smooth_price = st.slider("Price smoothing", 0.0, 8.0, 4.0, step=0.5)
    smooth_breadth = st.slider("Breadth smoothing", 0.0, 8.0, 4.0, step=0.5)
    breadth_weight = st.slider("Breadth weight", 0.0, 1.0, 0.70, step=0.05)
    stop_loss_pct = st.slider("Stop loss %", 0.02, 0.12, 0.06, step=0.01)
    target_vol = st.slider("Target vol", 0.05, 0.25, 0.15, step=0.01)

    st.header("View")
    view_mode = st.selectbox("Oscillator view", ["Combined", "Breadth", "Price", "All"], index=0)
    chart_window = st.selectbox("Chart window", ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "Max"], index=2)

    run = st.button("Build oscillator", type="primary", use_container_width=True)

st.info("Nothing heavy runs until you click **Build oscillator**.")

if not run:
    st.stop()

try:
    uploaded_price_bytes = price_file.getvalue() if price_file is not None else None
    uploaded_price_name = price_file.name if price_file is not None else None
    uploaded_zip_bytes = breadth_zip.getvalue() if breadth_zip is not None else None

    with st.spinner("Building oscillator..."):
        osc_df, backtest, summary = build_model(
            price_mode=price_mode,
            symbol=symbol,
            period=period,
            uploaded_price_bytes=uploaded_price_bytes,
            uploaded_price_name=uploaded_price_name,
            uploaded_zip_bytes=uploaded_zip_bytes,
            lookback=lookback,
            smooth_price=smooth_price,
            smooth_breadth=smooth_breadth,
            breadth_weight=breadth_weight,
            stop_loss_pct=stop_loss_pct,
            target_vol=target_vol,
        )

    if osc_df.empty:
        st.error("No model output was produced.")
        st.stop()

    bars = bars_from_window(chart_window)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("State", str(summary.get("state", "N/A")))
    c2.metric("Regime", str(summary.get("regime", "N/A")))
    c3.metric("Final Osc", f"{summary.get('final_osc', float('nan')):.2f}")
    c4.metric("Quality", f"{summary.get('quality_score', float('nan')):.2f}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Thrust", f"{summary.get('thrust_score', float('nan')):.2f}")
    d2.metric("Long Enter", f"{summary.get('long_enter_threshold', float('nan')):.2f}")
    d3.metric("Long Exit", f"{summary.get('long_exit_threshold', float('nan')):.2f}")

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Backtest", "Data"])

    with tab1:
        st.plotly_chart(make_oscillator_chart(osc_df, view_mode=view_mode, lookback_bars=bars), use_container_width=True)
        st.plotly_chart(make_price_chart(osc_df, lookback_bars=bars), use_container_width=True)

    with tab2:
        metrics = backtest.get("metrics", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Return", f"{metrics.get('total_return', float('nan')):.2%}" if metrics else "N/A")
        m2.metric("CAGR", f"{metrics.get('cagr', float('nan')):.2%}" if metrics else "N/A")
        m3.metric("Sharpe", f"{metrics.get('sharpe', float('nan')):.2f}" if metrics else "N/A")
        m4.metric("Max DD", f"{metrics.get('max_drawdown', float('nan')):.2%}" if metrics else "N/A")
        m5.metric("Trades", str(metrics.get("num_trades", 0)) if metrics else "0")

        st.plotly_chart(make_equity_chart(backtest), use_container_width=True)

        trades = backtest.get("trades", pd.DataFrame())
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            st.subheader("Trades")
            st.dataframe(trades, use_container_width=True)
        else:
            st.caption("No trades generated with the current settings.")

    with tab3:
        st.subheader("Latest rows")
        st.dataframe(osc_df.tail(50), use_container_width=True)

except Exception as e:
    st.error(f"App failed: {e}")
    st.code(traceback.format_exc())
