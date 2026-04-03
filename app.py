
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.offsets import BDay
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

st.set_page_config(page_title="Consensus Bell Curve Oscillator v3", layout="wide")

STATE_LADDER = [
    (0, 10, "Washout", "#7f1d1d"),
    (10, 20, "Bounce", "#b45309"),
    (20, 40, "Repair", "#ca8a04"),
    (40, 60, "Neutral", "#475569"),
    (60, 80, "Bull Thrust", "#15803d"),
    (80, 100, "Exhaustion", "#1d4ed8"),
]

DATA_DIR = Path("app_state")
DATA_DIR.mkdir(exist_ok=True)
SNAPSHOT_LOG_PATH = DATA_DIR / "consensus_snapshot_log_v3.csv"


@dataclass
class SeriesConfig:
    series_weights: Dict[str, float] = field(default_factory=lambda: {
        "SPXA50R": 1.50,
        "BPSPX": 1.40,
        "BPNYA": 1.10,
        "NYMO": 1.20,
        "NYSI": 1.05,
        "NYAD": 1.00,
        "SPXADP": 0.90,
        "OEXA50R": 0.80,
        "CPCE": 0.55,
        "TRIN": 0.55,
    })
    cci_weight: float = 0.50
    tsi_weight: float = 0.30
    bbp_weight: float = 0.20


def clamp01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(lower=0.0, upper=1.0)


def serialize_series(series: pd.Series) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    idx = tuple(pd.Index(series.index).strftime("%Y-%m-%d"))
    vals = tuple(float(x) if pd.notna(x) else np.nan for x in pd.to_numeric(series, errors="coerce").values)
    return idx, vals


def deserialize_series(payload: Tuple[Tuple[str, ...], Tuple[float, ...]]) -> pd.Series:
    idx, vals = payload
    return pd.Series(vals, index=pd.to_datetime(list(idx))).sort_index()


def parse_stockcharts_history_text(text: str) -> pd.Series:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return pd.Series(dtype=float)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return pd.Series(dtype=float)
    rows = []
    for line in lines[2:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        dt = pd.to_datetime(parts[0], errors="coerce")
        close_val = pd.to_numeric(parts[4], errors="coerce")
        if pd.notna(dt) and pd.notna(close_val):
            rows.append((dt, float(close_val)))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({dt: val for dt, val in rows}).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.astype(float)


@st.cache_data(show_spinner=False)
def load_zip_bundle_cached(zip_bytes: bytes) -> Dict[str, Tuple[Tuple[str, ...], Tuple[float, ...]]]:
    bundle: Dict[str, Tuple[Tuple[str, ...], Tuple[float, ...]]] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            stem = name.split("/")[-1].replace(".csv", "")
            text = zf.read(name).decode("utf-8", errors="ignore")
            s = parse_stockcharts_history_text(text)
            if not s.empty:
                bundle[stem] = serialize_series(s)
    return bundle


def load_zip_bundle(zip_bytes: bytes) -> Dict[str, pd.Series]:
    raw = load_zip_bundle_cached(zip_bytes)
    return {k: deserialize_series(v) for k, v in raw.items()}


def parse_snapshot_csv(file_bytes: bytes) -> Dict[str, float]:
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols = {c.lower().strip(): c for c in df.columns}
    if "symbol" not in cols or "close" not in cols:
        raise ValueError("Snapshot CSV must contain Symbol and Close columns.")
    sym_col = cols["symbol"]
    close_col = cols["close"]
    tmp = df[[sym_col, close_col]].copy()
    tmp[sym_col] = tmp[sym_col].astype(str).str.strip().str.upper()
    tmp[close_col] = pd.to_numeric(tmp[close_col], errors="coerce")
    tmp = tmp.dropna()
    out: Dict[str, float] = {}
    for _, row in tmp.iterrows():
        out[str(row[sym_col])] = float(row[close_col])
    return out


def parse_sequence_number(filename: str) -> Optional[int]:
    m = re.search(r"\((\d+)\)", filename)
    return int(m.group(1)) if m else None


def next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d) + BDay(1)


def snapshot_value(snapshot_map: Dict[str, float], *aliases: str) -> Optional[float]:
    for alias in aliases:
        key = alias.upper()
        if key in snapshot_map:
            return snapshot_map[key]
    return None


def stem_to_display_name(stem: str) -> str:
    mapping = {
        "_spxa50r": "SPXA50R",
        "_Bpspx": "BPSPX",
        "_Bpnya": "BPNYA",
        "_nymo": "NYMO",
        "_nySI": "NYSI",
        "_nyad": "NYAD",
        "_trin": "TRIN",
        "_cpce": "CPCE",
        "_cpc": "CPCE",
        "_oexa50r": "OEXA50R",
        "_spxadp": "SPXADP",
    }
    return mapping.get(stem, stem)


def display_name_to_snapshot_aliases(name: str) -> Tuple[str, ...]:
    mapping = {
        "SPXA50R": ("$SPXA50R", "SPXA50R"),
        "BPSPX": ("$BPSPX", "BPSPX"),
        "BPNYA": ("$BPNYA", "BPNYA"),
        "NYMO": ("$NYMO", "NYMO"),
        "NYSI": ("$NYSI", "NYSI"),
        "NYAD": ("$NYAD", "NYAD"),
        "TRIN": ("$TRIN", "TRIN"),
        "CPCE": ("$CPCE", "$CPC", "CPCE", "CPC"),
        "OEXA50R": ("$OEXA50R", "OEXA50R"),
        "SPXADP": ("$SPXADP", "SPXADP"),
    }
    return mapping.get(name, (name,))


def normalize_for_direction(series: pd.Series, name: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").sort_index().copy()
    if name in {"TRIN", "CPCE"}:
        x = -x
    elif name == "NYAD":
        x = x.pct_change(5)
    return x


def cci(series: pd.Series, length: int = 20) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    ma = x.rolling(length, min_periods=max(5, length // 2)).mean()
    md = (x - ma).abs().rolling(length, min_periods=max(5, length // 2)).mean().clip(lower=1e-8)
    return (x - ma) / (0.015 * md)


def tsi(series: pd.Series, fast: int = 25, slow: int = 13, signal: int = 7) -> Tuple[pd.Series, pd.Series]:
    x = pd.to_numeric(series, errors="coerce")
    mom = x.diff()
    a = mom.ewm(span=fast, adjust=False).mean().ewm(span=slow, adjust=False).mean()
    b = mom.abs().ewm(span=fast, adjust=False).mean().ewm(span=slow, adjust=False).mean().clip(lower=1e-8)
    tsi_line = 100.0 * (a / b)
    tsi_signal = tsi_line.ewm(span=signal, adjust=False).mean()
    return tsi_line, tsi_signal


def bbp(series: pd.Series, length: int = 20, stds: float = 2.0) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    ma = x.rolling(length, min_periods=max(5, length // 2)).mean()
    sd = x.rolling(length, min_periods=max(5, length // 2)).std()
    upper = ma + stds * sd
    lower = ma - stds * sd
    width = (upper - lower).replace(0, np.nan)
    return (x - lower) / width


def bell_curve_transform(series: pd.Series, lookback: int = 100, sigma: float = 2.0, z_clip: float = 3.5) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").copy()
    x = x.interpolate(limit_direction="both").ffill().bfill()
    min_periods = max(20, int(lookback * 0.67))
    mean_ = x.rolling(lookback, min_periods=min_periods).mean()
    std_ = x.rolling(lookback, min_periods=min_periods).std().clip(lower=1e-8)
    z = ((x - mean_) / std_).clip(-z_clip, z_clip)
    cdf = pd.Series(norm.cdf(z) * 100.0, index=x.index)
    arr = cdf.fillna(50.0).to_numpy(dtype=float)
    out = pd.Series(gaussian_filter1d(arr, sigma=sigma), index=x.index)
    out[mean_.isna()] = np.nan
    return out


def cci_to_prob(cci_line: pd.Series) -> pd.Series:
    return clamp01((cci_line + 200.0) / 400.0)


def tsi_to_prob(tsi_line: pd.Series, tsi_signal: pd.Series) -> pd.Series:
    level = clamp01((tsi_line + 25.0) / 50.0)
    spread = tsi_line - tsi_signal
    hook = clamp01((spread + 10.0) / 20.0)
    return clamp01(0.65 * level + 0.35 * hook)


def bbp_to_prob(bbp_line: pd.Series) -> pd.Series:
    return clamp01(bbp_line)


def timeframe_params(timeframe: str) -> Tuple[int, int, int, int, int, int, float]:
    if timeframe == "Weekly":
        return 14, 13, 7, 7, 20, 52, 2.6
    if timeframe == "Hourly":
        return 14, 12, 6, 6, 20, 63, 1.6
    return 20, 25, 13, 7, 20, 84, 2.0


def calculate_individual_consensus(series: pd.Series, cfg: SeriesConfig, timeframe: str) -> pd.DataFrame:
    x = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    cci_len, tsi_fast, tsi_slow, tsi_sig, bb_len, final_lookback, final_sigma = timeframe_params(timeframe)

    cci_line = cci(x, cci_len)
    tsi_line, tsi_signal = tsi(x, tsi_fast, tsi_slow, tsi_sig)
    bbp_line = bbp(x, bb_len)

    cci_prob = cci_to_prob(cci_line)
    tsi_prob = tsi_to_prob(tsi_line, tsi_signal)
    bbp_prob = bbp_to_prob(bbp_line)

    raw = (
        cfg.cci_weight * cci_prob +
        cfg.tsi_weight * tsi_prob +
        cfg.bbp_weight * bbp_prob
    )
    osc = bell_curve_transform(raw, lookback=final_lookback, sigma=final_sigma)

    return pd.DataFrame({
        "raw_prob": raw,
        "osc": osc,
        "cci_prob": cci_prob,
        "tsi_prob": tsi_prob,
        "bbp_prob": bbp_prob,
    })


@st.cache_data(show_spinner=False)
def build_daily_consensus_cached(
    zip_bytes: bytes,
    snapshot_payload_json: str,
    anchor_date_str: str,
    cfg_dict: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    cfg = SeriesConfig(**cfg_dict)
    base_bundle = load_zip_bundle(zip_bytes)

    snapshot_payload = json.loads(snapshot_payload_json)
    snapshot_entries = []
    ordered = sorted(snapshot_payload, key=lambda x: (x["sequence"], x["filename"]))
    for i, info in enumerate(ordered):
        snap_date = pd.Timestamp(anchor_date_str) if i == 0 else next_business_day(snapshot_entries[-1]["assigned_date"])
        snap_map = info["snapshot_map"]
        snapshot_entries.append({
            "filename": info["filename"],
            "sequence": info["sequence"],
            "assigned_date": snap_date,
            "snapshot_map": snap_map,
        })

    stem_alias_map = {
        "_spxa50r": display_name_to_snapshot_aliases("SPXA50R"),
        "_Bpspx": display_name_to_snapshot_aliases("BPSPX"),
        "_Bpnya": display_name_to_snapshot_aliases("BPNYA"),
        "_nymo": display_name_to_snapshot_aliases("NYMO"),
        "_nySI": display_name_to_snapshot_aliases("NYSI"),
        "_nyad": display_name_to_snapshot_aliases("NYAD"),
        "_trin": display_name_to_snapshot_aliases("TRIN"),
        "_cpce": display_name_to_snapshot_aliases("CPCE"),
        "_cpc": display_name_to_snapshot_aliases("CPCE"),
        "_oexa50r": display_name_to_snapshot_aliases("OEXA50R"),
        "_spxadp": display_name_to_snapshot_aliases("SPXADP"),
    }

    updated = {k: v.copy() for k, v in base_bundle.items()}
    for entry in snapshot_entries:
        snap_date = pd.Timestamp(entry["assigned_date"])
        snap_map = entry["snapshot_map"]
        for stem, aliases in stem_alias_map.items():
            if stem in updated:
                val = snapshot_value(snap_map, *aliases)
                if val is not None:
                    updated[stem].loc[snap_date] = float(val)
                    updated[stem] = updated[stem].sort_index()
                    updated[stem] = updated[stem][~updated[stem].index.duplicated(keep="last")]

    processed = {}
    for stem, ser in updated.items():
        name = stem_to_display_name(stem)
        if name not in cfg.series_weights:
            continue
        base = normalize_for_direction(ser, name)
        if len(base.dropna()) < 30:
            continue
        processed[name] = calculate_individual_consensus(base, cfg, "Daily")

    if not processed:
        return pd.DataFrame(), pd.DataFrame(), json.dumps([])

    weights = pd.Series({name: cfg.series_weights[name] for name in processed.keys()}, dtype=float)

    raw_df = pd.concat([df["raw_prob"].rename(name) for name, df in processed.items()], axis=1).sort_index()
    cci_df = pd.concat([df["cci_prob"].rename(name) for name, df in processed.items()], axis=1).sort_index()
    tsi_df = pd.concat([df["tsi_prob"].rename(name) for name, df in processed.items()], axis=1).sort_index()
    bbp_df = pd.concat([df["bbp_prob"].rename(name) for name, df in processed.items()], axis=1).sort_index()

    weighted_raw = raw_df.multiply(weights, axis=1).sum(axis=1, min_count=1) / raw_df.notna().multiply(weights, axis=1).sum(axis=1).replace(0, np.nan)
    cci_cons = cci_df.multiply(weights, axis=1).sum(axis=1, min_count=1) / cci_df.notna().multiply(weights, axis=1).sum(axis=1).replace(0, np.nan)
    tsi_cons = tsi_df.multiply(weights, axis=1).sum(axis=1, min_count=1) / tsi_df.notna().multiply(weights, axis=1).sum(axis=1).replace(0, np.nan)
    bbp_cons = bbp_df.multiply(weights, axis=1).sum(axis=1, min_count=1) / bbp_df.notna().multiply(weights, axis=1).sum(axis=1).replace(0, np.nan)

    daily_summary = pd.DataFrame({
        "raw_prob": weighted_raw,
        "consensus_osc": bell_curve_transform(weighted_raw, lookback=84, sigma=2.0),
        "cci_prob_consensus": cci_cons * 100.0,
        "tsi_prob_consensus": tsi_cons * 100.0,
        "bbp_prob_consensus": bbp_cons * 100.0,
    }).sort_index()

    stamped_rows = []
    for entry in snapshot_entries:
        dt = pd.Timestamp(entry["assigned_date"])
        if len(daily_summary.dropna()) == 0:
            continue
        if dt in daily_summary.index:
            row = daily_summary.loc[dt]
        else:
            pos = daily_summary.index.get_indexer([dt], method="nearest")[0]
            row = daily_summary.iloc[pos]
        stamped_rows.append({
            "filename": entry["filename"],
            "sequence": entry["sequence"],
            "assigned_date": str(dt.date()),
            "consensus_osc": round(float(row["consensus_osc"]), 2),
            "cci_layer": round(float(row["cci_prob_consensus"]), 2),
            "tsi_layer": round(float(row["tsi_prob_consensus"]), 2),
            "bbp_layer": round(float(row["bbp_prob_consensus"]), 2),
        })
    stamped_df = pd.DataFrame(stamped_rows)
    entries_json = json.dumps([
        {
            "filename": e["filename"],
            "sequence": e["sequence"],
            "assigned_date": str(pd.Timestamp(e["assigned_date"]).date()),
            "symbol_count": len(e["snapshot_map"]),
        }
        for e in snapshot_entries
    ])
    return daily_summary, stamped_df, entries_json


@st.cache_data(show_spinner=False)
def derive_weekly_from_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    weekly = daily_df.resample("W-FRI").last().dropna(how="all")
    weekly["consensus_osc"] = bell_curve_transform(weekly["raw_prob"], lookback=39, sigma=2.5)
    return weekly


@st.cache_data(show_spinner=False)
def derive_hourly_like_from_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    hourly = daily_df.copy()
    hourly["consensus_osc"] = bell_curve_transform(hourly["raw_prob"], lookback=63, sigma=1.4)
    hourly["cci_prob_consensus"] = bell_curve_transform(hourly["cci_prob_consensus"], lookback=63, sigma=1.1)
    hourly["tsi_prob_consensus"] = bell_curve_transform(hourly["tsi_prob_consensus"], lookback=63, sigma=1.1)
    hourly["bbp_prob_consensus"] = bell_curve_transform(hourly["bbp_prob_consensus"], lookback=63, sigma=1.1)
    return hourly


def ladder_state(score: float) -> Tuple[str, str]:
    s = float(np.clip(score, 0.0, 100.0))
    for low, high, label, color in STATE_LADDER:
        if (low <= s < high) or (high == 100 and s <= high):
            return label, color
    return "Unknown", "#475569"


def bars_from_label(label: str) -> int:
    return {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "10Y": 2520, "Max": 0}.get(label, 252)


def trim_df(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    if bars == 0 or len(df) <= bars:
        return df.copy()
    return df.iloc[-bars:].copy()


def append_snapshot_log(entries_df: pd.DataFrame):
    if entries_df.empty:
        return
    if SNAPSHOT_LOG_PATH.exists():
        old = pd.read_csv(SNAPSHOT_LOG_PATH)
        combo = pd.concat([old, entries_df], ignore_index=True).drop_duplicates(subset=["filename", "assigned_date"], keep="last")
    else:
        combo = entries_df.copy()
    combo.to_csv(SNAPSHOT_LOG_PATH, index=False)


def load_snapshot_log() -> pd.DataFrame:
    if SNAPSHOT_LOG_PATH.exists():
        return pd.read_csv(SNAPSHOT_LOG_PATH)
    return pd.DataFrame(columns=["filename", "sequence", "assigned_date", "symbol_count"])


def render_state_ladder(score: float):
    active_label, _ = ladder_state(score)
    html_parts = []
    for low, high, label, color in STATE_LADDER:
        active = label == active_label
        border = "3px solid #111827" if active else "1px solid rgba(15,23,42,0.15)"
        shadow = "box-shadow: inset 0 0 0 9999px rgba(255,255,255,0.08);" if active else ""
        html_parts.append(
            f"""
            <div style="flex:1; min-width:120px; background:{color}; color:white; border:{border};
                        border-radius:12px; padding:10px 12px; {shadow}">
                <div style="font-size:12px; opacity:0.9;">{low:.0f}–{high:.0f}</div>
                <div style="font-size:16px; font-weight:700; line-height:1.2;">{label}</div>
            </div>
            """
        )
    st.markdown(
        f"""
        <div style="margin: 0.35rem 0 0.75rem 0;">
            <div style="font-size:0.95rem; font-weight:700; margin-bottom:0.35rem;">
                State ladder — current zone: {active_label} ({score:.2f})
            </div>
            <div style="display:flex; gap:8px; flex-wrap:wrap;">
                {''.join(html_parts)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def consensus_chart(df: pd.DataFrame, bars: int, timeframe: str, snapshot_entries_df: pd.DataFrame) -> go.Figure:
    plot_df = trim_df(df, bars)
    fig = go.Figure()

    for low, high, label, color in STATE_LADDER:
        fig.add_hrect(y0=low, y1=high, opacity=0.08, line_width=0, fillcolor=color)
        fig.add_annotation(
            xref="paper", x=1.0, y=(low + high) / 2.0,
            text=label, showarrow=False, xanchor="left", font=dict(size=11, color=color)
        )

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["consensus_osc"], mode="lines", name=f"{timeframe} Consensus Osc", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["cci_prob_consensus"], mode="lines", name="CCI layer", line=dict(width=1.8, dash="dash")))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["tsi_prob_consensus"], mode="lines", name="TSI layer", line=dict(width=1.8, dash="dot")))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bbp_prob_consensus"], mode="lines", name="%B layer", line=dict(width=1.8, dash="longdash")))

    if not plot_df.empty:
        fig.add_trace(go.Scatter(
            x=[plot_df.index[-1]], y=[float(plot_df["consensus_osc"].iloc[-1])],
            mode="markers", marker=dict(size=11, symbol="diamond"), name="Current"
        ))

    if not snapshot_entries_df.empty:
        for _, entry in snapshot_entries_df.iterrows():
            dt = pd.Timestamp(entry["assigned_date"])
            if len(plot_df) == 0 or dt < plot_df.index.min() or dt > plot_df.index.max():
                continue
            nearest_idx = plot_df.index.get_indexer([dt], method="nearest")[0]
            y = float(plot_df["consensus_osc"].iloc[nearest_idx])
            label = f"{dt.date()} · SC {entry['sequence']}"
            fig.add_vline(x=dt, line_dash="dot", line_width=1, opacity=0.45)
            fig.add_annotation(
                x=dt, y=y, text=label, showarrow=True, arrowhead=2,
                ay=-35, font=dict(size=10), bgcolor="rgba(255,255,255,0.85)"
            )

    fig.update_layout(
        title=f"{timeframe} Consensus Bell Curve Oscillator v3",
        height=560,
        yaxis_title="0–100 Score",
        yaxis=dict(range=[0, 100]),
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h")
    )
    return fig


st.title("Consensus Bell Curve Oscillator v3")
st.caption("Faster version: cache-heavy, daily-first build, then derive hourly/weekly from daily. Daily uploads are timestamped onto charts and tables.")

with st.sidebar:
    st.header("Inputs")
    breadth_zip = st.file_uploader("Historical breadth ZIP", type=["zip"])
    snapshot_files = st.file_uploader("Daily snapshot CSV files", type=["csv"], accept_multiple_files=True)

    st.header("Snapshot Date Stamping")
    anchor_date = st.date_input("First uploaded snapshot date", value=pd.Timestamp("2026-03-30").date())
    st.caption("Files are sorted by SC sequence number, then assigned sequential business days from this first date.")

    st.header("View")
    chart_window = st.selectbox("Chart window", ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "Max"], index=2)
    persist_log = st.checkbox("Save snapshot log to app_state", value=False)

    st.header("Performance")
    compact_mode = st.checkbox("Compact mode (lighter tables)", value=True)

if breadth_zip is None:
    st.info("Upload your historical ZIP first, then your daily snapshot CSV file(s).")
    st.stop()

cfg = SeriesConfig()
snapshot_payload = []
if snapshot_files:
    for f in snapshot_files:
        seq = parse_sequence_number(f.name)
        snapshot_payload.append({
            "filename": f.name,
            "sequence": seq if seq is not None else 999999,
            "snapshot_map": parse_snapshot_csv(f.getvalue()),
        })

daily_df, stamped_df, entries_json = build_daily_consensus_cached(
    breadth_zip.getvalue(),
    json.dumps(snapshot_payload, sort_keys=True),
    str(pd.Timestamp(anchor_date).date()),
    {
        "series_weights": cfg.series_weights,
        "cci_weight": cfg.cci_weight,
        "tsi_weight": cfg.tsi_weight,
        "bbp_weight": cfg.bbp_weight,
    },
)

if daily_df.empty:
    st.error("No usable series were found in the ZIP after normalization.")
    st.stop()

entries_df = pd.DataFrame(json.loads(entries_json))
if persist_log and not entries_df.empty:
    append_snapshot_log(entries_df)

weekly_df = derive_weekly_from_daily(daily_df)
hourly_df = derive_hourly_like_from_daily(daily_df)

tabs = st.tabs(["Hourly", "Daily", "Weekly", "Snapshot Log", "Notes"])
bars = bars_from_label(chart_window)

for tf, df, tab in [
    ("Hourly", hourly_df, tabs[0]),
    ("Daily", daily_df, tabs[1]),
    ("Weekly", weekly_df, tabs[2]),
]:
    with tab:
        last = df.dropna().iloc[-1]
        state, _ = ladder_state(float(last["consensus_osc"]))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Consensus Osc", f"{float(last['consensus_osc']):.2f}")
        c2.metric("State", state)
        c3.metric("CCI Layer", f"{float(last['cci_prob_consensus']):.1f}")
        c4.metric("TSI Layer", f"{float(last['tsi_prob_consensus']):.1f}")
        c5.metric("%B Layer", f"{float(last['bbp_prob_consensus']):.1f}")

        if not entries_df.empty:
            latest_snap = pd.to_datetime(entries_df["assigned_date"]).max()
            st.caption(f"Latest stamped upload date in model: **{latest_snap.date()}**")
        else:
            st.caption("No daily snapshot uploads stamped yet.")

        render_state_ladder(float(last["consensus_osc"]))
        st.plotly_chart(consensus_chart(df, bars, tf, entries_df), use_container_width=True)

        if not stamped_df.empty and tf in {"Hourly", "Daily", "Weekly"}:
            st.subheader(f"{tf} stamped upload readings")
            local = stamped_df.copy()
            if tf == "Weekly":
                temp = []
                for _, row in local.iterrows():
                    dt = pd.Timestamp(row["assigned_date"])
                    if len(weekly_df.dropna()) == 0:
                        continue
                    pos = weekly_df.index.get_indexer([dt], method="nearest")[0]
                    r = weekly_df.iloc[pos]
                    temp.append({
                        "filename": row["filename"],
                        "sequence": row["sequence"],
                        "assigned_date": row["assigned_date"],
                        "consensus_osc": round(float(r["consensus_osc"]), 2),
                        "cci_layer": round(float(r["cci_prob_consensus"]), 2),
                        "tsi_layer": round(float(r["tsi_prob_consensus"]), 2),
                        "bbp_layer": round(float(r["bbp_prob_consensus"]), 2),
                    })
                local = pd.DataFrame(temp)
            elif tf == "Hourly":
                temp = []
                for _, row in local.iterrows():
                    dt = pd.Timestamp(row["assigned_date"])
                    if len(hourly_df.dropna()) == 0:
                        continue
                    pos = hourly_df.index.get_indexer([dt], method="nearest")[0]
                    r = hourly_df.iloc[pos]
                    temp.append({
                        "filename": row["filename"],
                        "sequence": row["sequence"],
                        "assigned_date": row["assigned_date"],
                        "consensus_osc": round(float(r["consensus_osc"]), 2),
                        "cci_layer": round(float(r["cci_prob_consensus"]), 2),
                        "tsi_layer": round(float(r["tsi_prob_consensus"]), 2),
                        "bbp_layer": round(float(r["bbp_prob_consensus"]), 2),
                    })
                local = pd.DataFrame(temp)
            if compact_mode:
                st.dataframe(local.tail(12), use_container_width=True)
            else:
                st.dataframe(local, use_container_width=True)

with tabs[3]:
    st.subheader("Current upload assignment")
    st.dataframe(entries_df if not entries_df.empty else pd.DataFrame(columns=["filename", "sequence", "assigned_date", "symbol_count"]), use_container_width=True)
    st.subheader("Saved historical log")
    st.dataframe(load_snapshot_log(), use_container_width=True)

with tabs[4]:
    st.markdown(
        """
**What changed in v3**
- Builds the **daily consensus once**
- Derives **weekly** from daily resampling
- Derives **hourly-like timing** from the same daily raw consensus
- Adds `st.cache_data` around the expensive parts
- Keeps timestamped daily uploads on charts and in tables

**Core model**
- Per-series normalized oscillator first
- Then weighted breadth consensus
- CCI first, then TSI, then %B
        """
    )
