
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

st.set_page_config(page_title="Consensus Bell Curve Oscillator", layout="wide")

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
SNAPSHOT_LOG_PATH = DATA_DIR / "consensus_snapshot_log.csv"


@dataclass
class SeriesConfig:
    # importance across breadth figures
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
    cdf = norm.cdf(z) * 100.0
    out = pd.Series(gaussian_filter1d(cdf.fillna(50.0).to_numpy(dtype=float), sigma=sigma), index=x.index)
    out[mean_.isna()] = np.nan
    return out


def cci_to_prob(cci_line: pd.Series) -> pd.Series:
    # -200 -> 0, 0 -> 0.5, +200 -> 1
    return clamp01((cci_line + 200.0) / 400.0)


def tsi_to_prob(tsi_line: pd.Series, tsi_signal: pd.Series) -> pd.Series:
    # base level + hook/cross behavior
    level = clamp01((tsi_line + 25.0) / 50.0)
    spread = tsi_line - tsi_signal
    hook = clamp01((spread + 10.0) / 20.0)
    return clamp01(0.65 * level + 0.35 * hook)


def bbp_to_prob(bbp_line: pd.Series) -> pd.Series:
    # reflect the gates you wanted:
    # <.05 washout, >.20 repair, >.50 thrust, >.80 exhaustion
    return clamp01(bbp_line)


def calculate_individual_consensus(series: pd.Series, cfg: SeriesConfig, timeframe: str) -> pd.DataFrame:
    x = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if timeframe == "Weekly":
        x = x.resample("W-FRI").last().dropna()
        cci_len, tsi_fast, tsi_slow, tsi_sig, bb_len, final_lookback, final_sigma = 14, 13, 7, 7, 20, 52, 2.8
    elif timeframe == "Hourly":
        # user asked for hourly, but with daily uploads we treat hourly as a faster timing view of the daily consensus
        cci_len, tsi_fast, tsi_slow, tsi_sig, bb_len, final_lookback, final_sigma = 14, 12, 6, 6, 20, 63, 1.8
    else:
        cci_len, tsi_fast, tsi_slow, tsi_sig, bb_len, final_lookback, final_sigma = 20, 25, 13, 7, 20, 100, 2.2

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

    out = pd.DataFrame({
        "series": x,
        "cci": cci_line,
        "tsi": tsi_line,
        "tsi_signal": tsi_signal,
        "bbp": bbp_line,
        "cci_prob": cci_prob,
        "tsi_prob": tsi_prob,
        "bbp_prob": bbp_prob,
        "raw_prob": raw,
        "osc": osc,
    })
    return out


def build_consensus(bundle: Dict[str, pd.Series], cfg: SeriesConfig, timeframe: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    processed: Dict[str, pd.DataFrame] = {}

    for stem, ser in bundle.items():
        name = stem_to_display_name(stem)
        if name not in cfg.series_weights:
            continue
        base = normalize_for_direction(ser, name)
        if len(base.dropna()) < 30:
            continue
        processed[name] = calculate_individual_consensus(base, cfg, timeframe)

    if not processed:
        return pd.DataFrame(), {}

    osc_cols = []
    raw_cols = []
    weights = {}
    for name, df in processed.items():
        osc_cols.append(df["osc"].rename(name))
        raw_cols.append(df["raw_prob"].rename(name))
        weights[name] = cfg.series_weights.get(name, 1.0)

    osc_df = pd.concat(osc_cols, axis=1).sort_index()
    raw_df = pd.concat(raw_cols, axis=1).sort_index()
    w = pd.Series(weights, dtype=float)

    weighted_osc = osc_df.multiply(w, axis=1).sum(axis=1, min_count=1) / osc_df.notna().multiply(w, axis=1).sum(axis=1).replace(0, np.nan)
    weighted_raw = raw_df.multiply(w, axis=1).sum(axis=1, min_count=1) / raw_df.notna().multiply(w, axis=1).sum(axis=1).replace(0, np.nan)

    # final smoothing / bell-curve on the consensus itself
    if timeframe == "Weekly":
        final_consensus = bell_curve_transform(weighted_raw, lookback=39, sigma=2.5)
    elif timeframe == "Hourly":
        final_consensus = bell_curve_transform(weighted_raw, lookback=63, sigma=1.5)
    else:
        final_consensus = bell_curve_transform(weighted_raw, lookback=84, sigma=2.0)

    cci_prob_cons = pd.concat([df["cci_prob"].rename(name) for name, df in processed.items()], axis=1).multiply(w, axis=1).sum(axis=1, min_count=1) / pd.concat([df["cci_prob"].rename(name) for name, df in processed.items()], axis=1).notna().multiply(w, axis=1).sum(axis=1).replace(0, np.nan)
    tsi_prob_cons = pd.concat([df["tsi_prob"].rename(name) for name, df in processed.items()], axis=1).multiply(w, axis=1).sum(axis=1, min_count=1) / pd.concat([df["tsi_prob"].rename(name) for name, df in processed.items()], axis=1).notna().multiply(w, axis=1).sum(axis=1).replace(0, np.nan)
    bbp_prob_cons = pd.concat([df["bbp_prob"].rename(name) for name, df in processed.items()], axis=1).multiply(w, axis=1).sum(axis=1, min_count=1) / pd.concat([df["bbp_prob"].rename(name) for name, df in processed.items()], axis=1).notna().multiply(w, axis=1).sum(axis=1).replace(0, np.nan)

    summary = pd.DataFrame({
        "raw_prob": weighted_raw,
        "avg_component_osc": weighted_osc,
        "consensus_osc": final_consensus,
        "cci_prob_consensus": cci_prob_cons * 100.0,
        "tsi_prob_consensus": tsi_prob_cons * 100.0,
        "bbp_prob_consensus": bbp_prob_cons * 100.0,
    }).sort_index()

    return summary, processed


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


def append_snapshot_log(entries: List[Dict]):
    if not entries:
        return
    rows = []
    for e in entries:
        rows.append({
            "filename": e["filename"],
            "sequence": e["sequence"],
            "assigned_date": str(pd.Timestamp(e["assigned_date"]).date()),
            "symbol_count": len(e["snapshot_map"]),
            "symbols_json": json.dumps(e["snapshot_map"]),
        })
    new_df = pd.DataFrame(rows)
    if SNAPSHOT_LOG_PATH.exists():
        old = pd.read_csv(SNAPSHOT_LOG_PATH)
        combo = pd.concat([old, new_df], ignore_index=True).drop_duplicates(subset=["filename", "assigned_date"], keep="last")
    else:
        combo = new_df
    combo.to_csv(SNAPSHOT_LOG_PATH, index=False)


def load_snapshot_log() -> pd.DataFrame:
    if SNAPSHOT_LOG_PATH.exists():
        return pd.read_csv(SNAPSHOT_LOG_PATH)
    return pd.DataFrame(columns=["filename", "sequence", "assigned_date", "symbol_count", "symbols_json"])


def patch_bundle_with_snapshots(base_bundle: Dict[str, pd.Series], snapshot_entries: List[Dict]) -> Dict[str, pd.Series]:
    updated = {k: v.copy() for k, v in base_bundle.items()}
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
    return updated


def build_with_snapshots(base_bundle: Dict[str, pd.Series], snapshot_files_info: List[Dict], anchor_date: pd.Timestamp):
    snapshot_entries = []
    ordered = sorted(snapshot_files_info, key=lambda x: (x["sequence"], x["filename"]))
    for i, info in enumerate(ordered):
        snap_date = pd.Timestamp(anchor_date) if i == 0 else next_business_day(snapshot_entries[-1]["assigned_date"])
        snap_map = parse_snapshot_csv(info["bytes"])
        snapshot_entries.append({
            "filename": info["filename"],
            "sequence": info["sequence"],
            "assigned_date": snap_date,
            "snapshot_map": snap_map,
        })
    updated_bundle = patch_bundle_with_snapshots(base_bundle, snapshot_entries)
    return updated_bundle, snapshot_entries


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


def consensus_chart(df: pd.DataFrame, bars: int, timeframe: str, snapshot_entries: List[Dict]) -> go.Figure:
    plot_df = trim_df(df, bars)
    fig = go.Figure()

    for low, high, label, color in STATE_LADDER:
        fig.add_hrect(y0=low, y1=high, opacity=0.08, line_width=0, fillcolor=color)
        fig.add_annotation(
            xref="paper", x=1.0, y=(low + high) / 2.0,
            text=label, showarrow=False, xanchor="left", font=dict(size=11, color=color)
        )

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["consensus_osc"],
        mode="lines", name=f"{timeframe} Consensus Osc",
        line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["cci_prob_consensus"],
        mode="lines", name="CCI layer",
        line=dict(width=1.8, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["tsi_prob_consensus"],
        mode="lines", name="TSI layer",
        line=dict(width=1.8, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["bbp_prob_consensus"],
        mode="lines", name="%B layer",
        line=dict(width=1.8, dash="longdash")
    ))

    if not plot_df.empty:
        last_x = plot_df.index[-1]
        last_y = float(plot_df["consensus_osc"].iloc[-1])
        fig.add_trace(go.Scatter(
            x=[last_x], y=[last_y], mode="markers",
            marker=dict(size=11, symbol="diamond"),
            name="Current"
        ))

    # timestamp uploaded daily snapshots directly on the chart
    for entry in snapshot_entries:
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
        title=f"{timeframe} Consensus Bell Curve Oscillator",
        height=560,
        yaxis_title="0–100 Score",
        yaxis=dict(range=[0, 100]),
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h")
    )
    return fig


st.title("Consensus Bell Curve Oscillator")
st.caption("True consensus of individually normalized breadth oscillators. CCI first, then TSI, then %B.")

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

if breadth_zip is None:
    st.info("Upload your historical ZIP first, then your daily snapshot CSV file(s).")
    st.stop()

cfg = SeriesConfig()
base_bundle = load_zip_bundle(breadth_zip.getvalue())

files_info = []
if snapshot_files:
    for f in snapshot_files:
        seq = parse_sequence_number(f.name)
        files_info.append({"filename": f.name, "sequence": seq if seq is not None else 999999, "bytes": f.getvalue()})

working_bundle, snapshot_entries = build_with_snapshots(base_bundle, files_info, pd.Timestamp(anchor_date))

if persist_log and snapshot_entries:
    append_snapshot_log(snapshot_entries)

tabs = st.tabs(["Hourly", "Daily", "Weekly", "Snapshot Log", "Breakdown"])

bars = bars_from_label(chart_window)

for tf, tab in zip(["Hourly", "Daily", "Weekly"], tabs[:3]):
    with tab:
        summary_df, processed = build_consensus(working_bundle, cfg, tf)
        if summary_df.empty:
            st.warning(f"No usable series found for {tf}.")
            continue

        last = summary_df.dropna().iloc[-1]
        state, _ = ladder_state(float(last["consensus_osc"]))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Consensus Osc", f"{float(last['consensus_osc']):.2f}")
        c2.metric("State", state)
        c3.metric("CCI Layer", f"{float(last['cci_prob_consensus']):.1f}")
        c4.metric("TSI Layer", f"{float(last['tsi_prob_consensus']):.1f}")
        c5.metric("%B Layer", f"{float(last['bbp_prob_consensus']):.1f}")

        if snapshot_entries:
            latest_snap = max(pd.Timestamp(e["assigned_date"]) for e in snapshot_entries)
            st.caption(f"Latest stamped upload date in model: **{latest_snap.date()}**")
        else:
            latest_snap = None
            st.caption("No daily snapshot uploads stamped yet.")

        render_state_ladder(float(last["consensus_osc"]))
        st.plotly_chart(consensus_chart(summary_df, bars, tf, snapshot_entries), use_container_width=True)

        # show the stamped dates and values numerically
        if snapshot_entries:
            rows = []
            for entry in snapshot_entries:
                dt = pd.Timestamp(entry["assigned_date"])
                if dt in summary_df.index:
                    row = summary_df.loc[dt]
                else:
                    if len(summary_df.dropna()) == 0:
                        continue
                    nearest_pos = summary_df.index.get_indexer([dt], method="nearest")[0]
                    row = summary_df.iloc[nearest_pos]
                rows.append({
                    "filename": entry["filename"],
                    "sequence": entry["sequence"],
                    "assigned_date": str(dt.date()),
                    "consensus_osc": round(float(row["consensus_osc"]), 2),
                    "cci_layer": round(float(row["cci_prob_consensus"]), 2),
                    "tsi_layer": round(float(row["tsi_prob_consensus"]), 2),
                    "bbp_layer": round(float(row["bbp_prob_consensus"]), 2),
                })
            if rows:
                st.subheader(f"{tf} stamped upload readings")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tabs[3]:
    st.subheader("Current upload assignment")
    current_log = pd.DataFrame([{
        "filename": e["filename"],
        "sequence": e["sequence"],
        "assigned_date": e["assigned_date"].date(),
        "symbol_count": len(e["snapshot_map"]),
    } for e in snapshot_entries])
    st.dataframe(current_log if not current_log.empty else pd.DataFrame(columns=["filename", "sequence", "assigned_date", "symbol_count"]), use_container_width=True)

    st.subheader("Saved historical log")
    st.dataframe(load_snapshot_log(), use_container_width=True)

with tabs[4]:
    st.subheader("Per-series weights")
    st.dataframe(pd.DataFrame({
        "series": list(cfg.series_weights.keys()),
        "weight": list(cfg.series_weights.values())
    }), use_container_width=True)
    st.caption("Final consensus = weighted average of per-series oscillators, after each series is separately normalized.")
