import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="Consensus Bell Curve Oscillator", layout="wide")

STATE_LADDER = [
    (0,10,"Washout","#7f1d1d"),
    (10,20,"Bounce","#b45309"),
    (20,40,"Repair","#ca8a04"),
    (40,60,"Neutral","#475569"),
    (60,80,"Bull Thrust","#15803d"),
    (80,100,"Exhaustion","#1d4ed8"),
]

def parse_stockcharts_csv(text):
    rows=[]
    lines=text.replace("\r","").split("\n")
    for line in lines[2:]:
        parts=line.split(",")
        if len(parts)<5: continue
        try:
            d=pd.to_datetime(parts[0])
            v=float(parts[4])
            rows.append((d,v))
        except: pass
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series({d:v for d,v in rows}).sort_index()

def load_zip(zip_bytes):
    out={}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if name.endswith(".csv"):
                txt=z.read(name).decode("utf-8","ignore")
                s = parse_stockcharts_csv(txt)
                if not s.empty:
                    out[name]=s
    return out

def CCI(series,n=20):
    ma=series.rolling(n).mean()
    md=(series-ma).abs().rolling(n).mean()
    return (series-ma)/(0.015*md)

def TSI(series,r=25,s=13):
    m=series.diff()
    a=m.ewm(span=r).mean().ewm(span=s).mean()
    b=m.abs().ewm(span=r).mean().ewm(span=s).mean()
    return 100*(a/b)

def BBP(series,n=20):
    ma=series.rolling(n).mean()
    sd=series.rolling(n).std()
    upper=ma+2*sd
    lower=ma-2*sd
    return (series-lower)/(upper-lower)

def bell_curve_transform(series,lookback=100,sigma=2):
    """Transforms a raw series into a 0-100 normalized score"""
    m=series.rolling(lookback).mean()
    s=series.rolling(lookback).std().clip(lower=1e-8)
    z=(series-m)/s
    cdf=norm.cdf(z)*100
    arr=gaussian_filter1d(cdf.fillna(50).values,sigma)
    return pd.Series(arr,index=series.index)

def calculate_individual_consensus(series):
    """Calculates CCI+TSI+BB consensus for a single series, then normalizes it"""
    cci=CCI(series)
    tsi=TSI(series)
    bb=BBP(series)
    
    # Normalize inputs to 0-1 range before weighting
    # CCI is typically -200 to 200
    cci_n=(cci+200)/400
    # TSI is typically -100 to 100
    tsi_n=(tsi+100)/200
    # BBP is already 0-1
    bb_n=bb
    
    raw=0.5*cci_n+0.3*tsi_n+0.2*bb_n
    
    # Apply Bell Curve to THIS SPECIFIC SERIES' consensus to normalize it 0-100
    # This allows us to compare Apples to Oranges later
    return bell_curve_transform(raw)

def ladder(score):
    for low,high,label,color in STATE_LADDER:
        if low<=score<high or (high==100 and score<=100):
            return label,color
    return "Unknown","#999"

def chart(series):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=series.index,y=series,name="Consensus Osc", line=dict(width=2.5, color='black')))
    
    # Add the colored background zones
    for low,high,label,color in STATE_LADDER:
        fig.add_hrect(y0=low,y1=high,fillcolor=color,opacity=0.1,line_width=0)
        
    # Add horizontal lines for the zones
    for low,high,label,color in STATE_LADDER:
        if low > 0:
            fig.add_hline(y=low, line_dash="dot", line_color="gray", opacity=0.3)
            
    fig.update_layout(height=500,yaxis_range=[0,100], title="Market Consensus Oscillator")
    return fig

st.title("Consensus Bell Curve Oscillator (Fixed)")
st.caption("Averages the normalized consensus (CCI+TSI+BB) of all uploaded charts.")

zip_file=st.file_uploader("Upload historical ZIP",type=["zip"])

if zip_file:
    bundle=load_zip(zip_file.getvalue())
    
    if not bundle:
        st.error("No valid CSV files found in ZIP.")
        st.stop()

    # FIX: Calculate consensus for EACH series first, then average the results
    individual_oscillators = []
    for name, series in bundle.items():
        # Calculate normalized 0-100 score for this specific ticker
        osc = calculate_individual_consensus(series)
        individual_oscillators.append(osc)
    
    # Combine them by averaging the 0-100 scores
    # This creates a true "Consensus of Probability"
    if individual_oscillators:
        combined_df = pd.concat(individual_oscillators, axis=1)
        # Average across all columns, skipping NaNs
        final_osc = combined_df.mean(axis=1, skipna=True)
    else:
        st.error("Could not process data.")
        st.stop()

    tf=st.radio("Timeframe",["Daily","Weekly"],horizontal=True)

    if tf=="Weekly":
        final_osc=final_osc.resample("W-FRI").last()

    # Apply one final smoothing to the consensus itself (optional, reduces noise further)
    final_osc_smooth = bell_curve_transform(final_osc, lookback=50, sigma=3)

    label,color=ladder(float(final_osc_smooth.iloc[-1]))

    c1,c2,c3 = st.columns(3)
    c1.metric("Current Score",round(float(final_osc_smooth.iloc[-1]),2))
    c2.metric("State",label)
    c3.metric("Charts Averaged", len(bundle))

    st.plotly_chart(chart(final_osc_smooth),use_container_width=True)

    st.write("### Input Breakdown")
    st.write(f"Aggregating signals from: {', '.join([k.split('/')[-1] for k in bundle.keys()])}")
