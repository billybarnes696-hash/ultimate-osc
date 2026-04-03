
import io
import zipfile
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from numbers import Real

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION / DATA CLASSES
# ============================================================================

@dataclass
class TradingCosts:
    commission: float = 0.0005
    slippage_bps: float = 2.0
    market_impact_bps: float = 5.0
    spread_bps: float = 1.0

    def __post_init__(self):
        for field_name in ["commission", "slippage_bps", "market_impact_bps", "spread_bps"]:
            val = getattr(self, field_name)
            if not isinstance(val, Real) or val < 0:
                raise ValueError(f"{field_name} must be a non-negative number, got {val}")

    def estimate_total_bps(self, position_size: float = 1.0) -> float:
        if position_size <= 0:
            return float("inf")
        return (
            self.slippage_bps
            + self.market_impact_bps * np.sqrt(position_size)
            + self.spread_bps
            + self.commission * 10000 / max(position_size, 1e-9)
        )


@dataclass
class RegimeThresholds:
    long_enter: float = 18.0
    long_exit: float = 45.0
    short_enter: float = 82.0
    short_exit: float = 55.0


@dataclass
class BacktestResult:
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    position: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# ROBUST NORMALIZATION
# ============================================================================

class BellCurveTransform:
    """
    Quantile-style oscillator on a 0-100 scale using rolling z-score -> Gaussian CDF.
    Improvements vs the original:
    - adaptive smoothing
    - clipping of z-score tails
    - explicit min history handling
    """
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

        rolling_mean = s.rolling(lookback, min_periods=max(20, int(lookback * min_frac))).mean()
        rolling_std = s.rolling(lookback, min_periods=max(20, int(lookback * min_frac))).std()

        z = (s - rolling_mean) / rolling_std.clip(lower=1e-8)
        z = z.clip(lower=-z_clip, upper=z_clip)

        cdf = norm.cdf(z) * 100.0

        if sigma > 0:
            smooth = gaussian_filter1d(cdf.to_numpy(dtype=float), sigma=sigma)
            out = pd.Series(smooth, index=series.index)
        else:
            out = pd.Series(cdf, index=series.index)

        out[rolling_mean.isna()] = np.nan
        return out


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_zscore(series: pd.Series, lookback: int = 63) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.rolling(lookback, min_periods=max(10, lookback // 3)).mean()
    sd = s.rolling(lookback, min_periods=max(10, lookback // 3)).std().clip(lower=1e-8)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def clip01(series: pd.Series, lo: float = 0.0, hi: float = 1.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=lo, upper=hi)


def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def roc(series: pd.Series, periods: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.pct_change(periods)


# ============================================================================
# REGIME DETECTOR
# ============================================================================

class RegimeDetector:
    """
    Improved regime model:
    - 252d trend
    - 63d trend
    - vol ratio
    - drawdown state
    """
    def detect_regime(self, price: pd.Series) -> pd.Series:
        px = pd.to_numeric(price, errors="coerce")
        rets = px.pct_change()

        trend_252 = px / px.shift(252) - 1.0
        trend_63 = px / px.shift(63) - 1.0

        vol_short = rets.rolling(20, min_periods=10).std() * np.sqrt(252)
        vol_long = rets.rolling(126, min_periods=60).std() * np.sqrt(252)
        vol_ratio = vol_short / vol_long.clip(lower=1e-8)

        dd = px / px.cummax() - 1.0

        f = pd.DataFrame(
            {
                "trend_252": trend_252,
                "trend_63": trend_63,
                "vol_ratio": vol_ratio,
                "drawdown": dd,
            }
        )

        regime = pd.Series("sideways", index=px.index, dtype=object)
        bull_mask = (
            (f["trend_252"] > 0.03)
            & (f["trend_63"] > -0.02)
            & (f["vol_ratio"] < 1.20)
            & (f["drawdown"] > -0.10)
        )
        bear_mask = (
            (f["trend_252"] < -0.03)
            | (f["trend_63"] < -0.06)
            | (f["vol_ratio"] > 1.35)
            | (f["drawdown"] < -0.15)
        )

        regime.loc[bull_mask] = "bull"
        regime.loc[bear_mask] = "bear"
        regime = regime.ffill().fillna("sideways")
        return regime

    def get_thresholds(self, regime: str) -> RegimeThresholds:
        if regime == "bull":
            return RegimeThresholds(long_enter=22.0, long_exit=48.0, short_enter=92.0, short_exit=60.0)
        if regime == "bear":
            return RegimeThresholds(long_enter=12.0, long_exit=35.0, short_enter=72.0, short_exit=50.0)
        return RegimeThresholds(long_enter=18.0, long_exit=45.0, short_enter=82.0, short_exit=55.0)


# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    def __init__(self, stop_loss_pct: float = 0.06, target_vol: float = 0.15):
        self.stop_loss_pct = stop_loss_pct
        self.target_vol = target_vol

    def volatility_scaled_size(self, current_vol: pd.Series) -> pd.Series:
        cv = pd.to_numeric(current_vol, errors="coerce").replace(0, np.nan)
        size = self.target_vol / cv
        size = size.clip(lower=0.25, upper=1.00)
        return size.fillna(1.0)


# ============================================================================
# BREADTH THRUST / QUALITY LAYER
# ============================================================================

class BreadthThrustDetector:
    """
    Added because this is one of the most useful short-term rally detectors.
    Works on the normalized breadth composite and selected participation series.
    """
    @staticmethod
    def calculate(
        breadth_osc: pd.Series,
        spxa50r: Optional[pd.Series] = None,
        bpspx: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        bo = pd.to_numeric(breadth_osc, errors="coerce")
        impulse_5 = bo.diff(5)
        impulse_10 = bo.diff(10)

        thrust = pd.Series(0.0, index=bo.index)

        thrust += (impulse_5 > 12).astype(float) * 0.40
        thrust += (impulse_10 > 18).astype(float) * 0.35
        thrust += ((bo > 35) & (bo.shift(3) < 20)).astype(float) * 0.25

        if spxa50r is not None:
            s50 = pd.to_numeric(spxa50r, errors="coerce")
            thrust += ((s50 > 30) & (s50.diff(3) > 2)).astype(float) * 0.25

        if bpspx is not None:
            bp = pd.to_numeric(bpspx, errors="coerce")
            thrust += ((bp.diff(5) > 1.5) | (bp > bp.rolling(20, min_periods=10).mean())).astype(float) * 0.15

        thrust = thrust.clip(lower=0.0, upper=1.0)
        return pd.DataFrame(
            {
                "thrust_score": thrust,
                "impulse_5": impulse_5,
                "impulse_10": impulse_10,
            }
        )


# ============================================================================
# ULTIMATE OSCILLATOR
# ============================================================================

class UltimateOscillator:
    """
    Major upgrades vs the original:
    1) lower lag smoothing
    2) weighted breadth inputs instead of equal weight
    3) optional SPXA200R support
    4) quality/thrust overlay
    5) price and breadth blended with explicit weights
    """

    def __init__(
        self,
        lookback: int = 126,
        smooth_price: float = 4.0,
        smooth_breadth: float = 4.0,
        breadth_weight: float = 0.70,
        price_weight: float = 0.30,
    ):
        self.lookback = lookback
        self.smooth_price = smooth_price
        self.smooth_breadth = smooth_breadth
        self.breadth_weight = breadth_weight
        self.price_weight = price_weight

        self.component_weights = {
            "_spxa50r": 2.2,
            "_spxa200r": 1.8,
            "_Bpspx": 1.6,
            "_Bpnya": 1.2,
            "_nymo": 1.3,
            "_nySI": 1.0,
            "_nyad": 0.9,
            "_trin": 0.7,
            "_cpc": 0.7,
        }

    def build_price_curve(self, price: pd.Series) -> pd.Series:
        return BellCurveTransform.calculate(
            price,
            lookback=self.lookback,
            sigma=self.smooth_price,
        )

    def _parse_sc_pipe_text(self, text: str) -> pd.Series:
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

    def load_zip_series(
        self,
        zip_bytes: bytes,
        snapshot_map: Optional[Dict[str, float]] = None,
        snap_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, pd.Series]:
        series_map: Dict[str, pd.Series] = {}

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            for stem in self.component_weights:
                matches = [n for n in names if stem.lower() in n.lower() and n.lower().endswith(".csv")]
                if not matches:
                    continue
                text = zf.read(matches[0]).decode("utf-8", errors="ignore")
                s = self._parse_sc_pipe_text(text)
                if s.empty:
                    continue

                if snapshot_map is not None and snap_date is not None and stem in snapshot_map:
                    s.loc[pd.Timestamp(snap_date)] = snapshot_map[stem]

                series_map[stem] = s.sort_index()

        return series_map

    def _transform_component(self, stem: str, s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").copy()

        if stem == "_nyad":
            x = roc(x, 5)  # less noisy than raw diff
        elif stem in ("_trin", "_cpc"):
            x = -x  # polarity inversion
        else:
            x = x

        return BellCurveTransform.calculate(
            x,
            lookback=self.lookback,
            sigma=self.smooth_breadth,
        )

    def build_breadth_curve(
        self,
        zip_bytes: Optional[bytes],
        snapshot_map: Optional[Dict[str, float]] = None,
        snap_date: Optional[pd.Timestamp] = None,
        return_details: bool = True,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        if zip_bytes is None:
            return pd.Series(dtype=float), pd.DataFrame()

        raw_map = self.load_zip_series(zip_bytes, snapshot_map=snapshot_map, snap_date=snap_date)
        transformed = {}
        for stem, s in raw_map.items():
            transformed[stem] = self._transform_component(stem, s)

        if not transformed:
            return pd.Series(dtype=float), pd.DataFrame()

        comp_df = pd.concat(transformed, axis=1).sort_index()

        weights = pd.Series(self.component_weights, dtype=float)
        common_cols = [c for c in comp_df.columns if c in weights.index]
        w = weights.loc[common_cols]

        weighted_sum = comp_df[common_cols].multiply(w, axis=1).sum(axis=1, min_count=1)
        weight_present = comp_df[common_cols].notna().multiply(w, axis=1).sum(axis=1).replace(0, np.nan)
        breadth_curve = weighted_sum / weight_present

        if return_details:
            detail_df = comp_df.copy()
            detail_df["breadth_curve"] = breadth_curve

            thrust_df = BreadthThrustDetector.calculate(
                breadth_curve,
                spxa50r=raw_map.get("_spxa50r"),
                bpspx=raw_map.get("_Bpspx"),
            )
            detail_df = detail_df.join(thrust_df, how="left")
            return breadth_curve, detail_df

        return breadth_curve, pd.DataFrame()

    def blend_final_oscillator(
        self,
        price: pd.Series,
        zip_bytes: Optional[bytes],
        snapshot_map: Optional[Dict[str, float]] = None,
        snap_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        price_curve = self.build_price_curve(price)
        breadth_curve, detail_df = self.build_breadth_curve(
            zip_bytes=zip_bytes,
            snapshot_map=snapshot_map,
            snap_date=snap_date,
            return_details=True,
        )

        df = pd.concat(
            {
                "price": pd.to_numeric(price, errors="coerce"),
                "price_curve": price_curve,
                "breadth_curve": breadth_curve,
            },
            axis=1,
        )

        if not detail_df.empty:
            df = df.join(detail_df, how="left")

        df["final_osc"] = (
            self.breadth_weight * df["breadth_curve"]
            + self.price_weight * df["price_curve"]
        )

        # Quality score: reward strong breadth, punish weak participation
        df["quality_score"] = 0.0
        if "_spxa50r" in df.columns:
            df["quality_score"] += ((df["_spxa50r"] > 30).astype(float) * 0.40)
            df["quality_score"] += ((df["_spxa50r"] > 40).astype(float) * 0.20)
        if "_Bpspx" in df.columns:
            df["quality_score"] += ((df["_Bpspx"] > 20).astype(float) * 0.20)
        if "thrust_score" in df.columns:
            df["quality_score"] += df["thrust_score"].fillna(0) * 0.20

        df["quality_score"] = df["quality_score"].clip(0, 1)
        return df


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class QuantBacktestEngine:
    """
    Long/flat engine using:
    - adaptive regime thresholds
    - quality gate
    - thrust confirmation
    - hysteresis (different enter/exit zones)
    """

    def __init__(
        self,
        costs: Optional[TradingCosts] = None,
        risk_mgr: Optional[RiskManager] = None,
        regime_detector: Optional[RegimeDetector] = None,
        min_quality_for_long: float = 0.45,
    ):
        self.costs = costs or TradingCosts()
        self.risk_mgr = risk_mgr or RiskManager()
        self.regime_detector = regime_detector or RegimeDetector()
        self.min_quality_for_long = min_quality_for_long

    @staticmethod
    def _performance_metrics(equity: pd.Series, returns: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
        if equity.empty:
            return {}

        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        ann_return = float((equity.iloc[-1] / equity.iloc[0]) ** (252 / max(len(equity), 1)) - 1.0)
        ann_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else np.nan
        sharpe = float(ann_return / ann_vol) if ann_vol and ann_vol > 0 else np.nan
        drawdown = equity / equity.cummax() - 1.0
        max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
        win_rate = float((trades["pnl"] > 0).mean()) if not trades.empty else np.nan
        return {
            "total_return": total_return,
            "annual_return": ann_return,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "trade_count": float(len(trades)),
            "win_rate": win_rate,
        }

    def run_backtest(self, df: pd.DataFrame) -> BacktestResult:
        work = df.copy()
        work["price"] = pd.to_numeric(work["price"], errors="coerce")
        work["final_osc"] = pd.to_numeric(work["final_osc"], errors="coerce")
        work["quality_score"] = pd.to_numeric(work.get("quality_score", 0.0), errors="coerce").fillna(0.0)
        work["thrust_score"] = pd.to_numeric(work.get("thrust_score", 0.0), errors="coerce").fillna(0.0)

        work = work.dropna(subset=["price", "final_osc"]).copy()
        if len(work) < 50:
            return BacktestResult()

        regime = self.regime_detector.detect_regime(work["price"])
        rets = work["price"].pct_change().fillna(0.0)
        vol = rets.rolling(20, min_periods=10).std() * np.sqrt(252)
        size = self.risk_mgr.volatility_scaled_size(vol)

        position = pd.Series(0.0, index=work.index)
        strat_returns = pd.Series(0.0, index=work.index)
        trades = []

        in_pos = False
        entry_px = np.nan
        entry_dt = None

        for i in range(1, len(work)):
            dt = work.index[i]
            prev_dt = work.index[i - 1]
            osc = float(work["final_osc"].iloc[i])
            q = float(work["quality_score"].iloc[i])
            thrust = float(work["thrust_score"].iloc[i])
            bar_regime = regime.iloc[i]

            th = self.regime_detector.get_thresholds(bar_regime)

            enter_long = (osc <= th.long_enter) and (q >= self.min_quality_for_long or thrust >= 0.60)
            exit_long = (osc >= th.long_exit) or (q < 0.20 and thrust < 0.20)

            if not in_pos and enter_long:
                in_pos = True
                entry_px = work["price"].iloc[i]
                entry_dt = dt
                trade_cost = self.costs.estimate_total_bps(position_size=float(size.iloc[i])) / 10000.0
                strat_returns.iloc[i] -= trade_cost

            elif in_pos and exit_long:
                exit_px = work["price"].iloc[i]
                pnl = exit_px / entry_px - 1.0
                trades.append(
                    {
                        "entry_date": entry_dt,
                        "exit_date": dt,
                        "entry_price": entry_px,
                        "exit_price": exit_px,
                        "pnl": pnl,
                        "regime_at_entry": regime.loc[entry_dt] if entry_dt in regime.index else np.nan,
                    }
                )
                in_pos = False
                entry_px = np.nan
                entry_dt = None
                trade_cost = self.costs.estimate_total_bps(position_size=float(size.iloc[i])) / 10000.0
                strat_returns.iloc[i] -= trade_cost

            position.iloc[i] = 1.0 if in_pos else 0.0
            strat_returns.iloc[i] += rets.iloc[i] * position.iloc[i] * float(size.iloc[i])

        equity = (1.0 + strat_returns.fillna(0.0)).cumprod()
        trades_df = pd.DataFrame(trades)
        metrics = self._performance_metrics(equity, strat_returns, trades_df)

        return BacktestResult(
            returns=strat_returns,
            equity_curve=equity,
            position=position,
            trades=trades_df,
            metrics=metrics,
        )


# ============================================================================
# SIGNAL SUMMARY
# ============================================================================

def summarize_current_signal(df: pd.DataFrame, regime_detector: Optional[RegimeDetector] = None) -> Dict[str, object]:
    if df.empty:
        return {}

    rd = regime_detector or RegimeDetector()
    regime = rd.detect_regime(df["price"])
    last_idx = df.index[-1]
    last_regime = regime.loc[last_idx]
    th = rd.get_thresholds(last_regime)

    osc = float(df["final_osc"].iloc[-1])
    q = float(df.get("quality_score", pd.Series(0.0, index=df.index)).iloc[-1])
    thrust = float(df.get("thrust_score", pd.Series(0.0, index=df.index)).iloc[-1])

    if osc <= th.long_enter and (q >= 0.45 or thrust >= 0.60):
        state = "LONG"
    elif osc >= th.short_enter:
        state = "RISK-OFF / TAKE PROFITS"
    else:
        state = "HOLD / NEUTRAL"

    return {
        "date": last_idx,
        "regime": last_regime,
        "state": state,
        "final_osc": osc,
        "quality_score": q,
        "thrust_score": thrust,
        "long_enter_threshold": th.long_enter,
        "long_exit_threshold": th.long_exit,
        "short_enter_threshold": th.short_enter,
        "short_exit_threshold": th.short_exit,
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example placeholder:
    # price = pd.read_csv("rsp.csv", parse_dates=["Date"]).set_index("Date")["Close"]
    # with open("breadth_history.zip", "rb") as f:
    #     zip_bytes = f.read()
    # uo = UltimateOscillator()
    # df = uo.blend_final_oscillator(price=price, zip_bytes=zip_bytes)
    # bt = QuantBacktestEngine().run_backtest(df)
    # print(bt.metrics)
    pass
