import streamlit as st
import sys, plotly.graph_objects as go, numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import helper
import analysis
import pandas as pd
st.set_page_config(page_title="Introduction", layout="wide")
st.title("Introduction - Signal Representation in Time Domain")

#Load data once
@st.cache_data
def load():
    df = helper.load_raw()
    return helper.get_driver(df)

driver_df = load()
stats     = analysis.compute_sampling_stats(driver_df)
lap_len   = analysis.compute_lap_lengths(driver_df)

#What is a signal? 
st.markdown(
    """
    ## What is a signal?
    A **signal** is any quantity that varies over an independent variable — here, time.
    F1 telemetry consists of multiple **discrete-time signals** sampled at a fixed rate
    by the car's data-acquisition system. Below is the raw Speed signal for a single lap:
    this is the most basic time-domain view.

    ## What time domain already tells us - a concrete example
    Before applying any transformation, the raw Speed signal alone reveals
    the full race strategy. The plot below is Lap 1 vs. the fastest lap:
    the stint structure, pit stops, and tyre effect are all visible
    without a single line of frequency analysis.
    """
)
fastest = analysis.get_fastest_lap(analysis.build_lap_summary(driver_df))
col1, col2 = st.columns(2)
with col1:
    fig1, _ = analysis.plot_speed_with_brakes(driver_df, lap=1)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Lap 1 — opening lap, cold tyres, fuel load at maximum")

with col2:
    fig2, _ = analysis.plot_speed_with_brakes(driver_df, lap=fastest)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        f"Lap {fastest} — fastest lap of the race, fresh soft tyres after 2nd pit stop"
    )

st.markdown(
    f"""
    The speed profile on Lap {fastest} shows visibly higher minimum corner speeds
    and sharper brake events compared to Lap 1 — this is tyre performance,
    fuel load reduction, and track evolution, all readable directly from the
    time domain signal.
    """
)
#Plot one raw lap to show what a "signal" looks like
lap1 = helper.get_lap(driver_df, helper.list_laps(driver_df)[10])
fig_intro = go.Figure()
fig_intro.add_trace(go.Scatter(
    x=lap1["Time"].dt.total_seconds().cumsum(),
    y=lap1["Speed"],
    mode="lines",
    line=dict(color="cornflowerblue", width=1.5),
    fill="tozeroy",
    fillcolor="rgba(100,149,237,0.1)",
    name="Speed",
))
fig_intro.update_layout(
    xaxis_title="Time within lap (s)",
    yaxis_title="Speed (km/h)",
    title=f"Raw Speed signal — Verstappen, Lap {helper.list_laps(driver_df)[10]}",
    height=320,
    margin=dict(l=60, r=20, t=50, b=60),
)
st.plotly_chart(fig_intro, use_container_width=True)
st.caption(
    "Each sample is one data point logged by the car. The x-axis is cumulative "
    "time; the y-axis is amplitude. Straights appear as plateaus, braking zones "
    "as sharp drops."
)

#Sampling rate
st.markdown("## Sampling rate of the dataset")
st.markdown(
    f"""
    Sampling rate is how many data points per second are recorded.
    A higher rate captures faster events (tyre lock-ups, kerb strikes).
    The Bahrain telemetry logs at approximately **{stats['mean_hz']} Hz** — one
    sample every **{stats['median_dt']} ms**.
    """
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean sampling rate", f"{stats['mean_hz']} Hz")
col2.metric("Median interval", f"{stats['median_dt']} ms")
col3.metric("Total samples (VER)", f"{stats['n_samples']:,}")
col4.metric("Laps recorded", stats["n_laps"])

#Variable-length laps
st.markdown(
    """
    ## Why laps have different lengths
    Because the logging is time-based, a slower lap (safety car, pit lap)
    accumulates **more** samples than a fast push lap.
    This is the key challenge for multi-lap comparison addressed in Methods.
    """
)

fig_len = go.Figure(go.Bar(
    x=lap_len["Lap"],
    y=lap_len["n_samples"],
    marker_color="cornflowerblue",
    hovertemplate="Lap %{x}: %{y} samples<extra></extra>",
))
fig_len.update_layout(
    xaxis_title="Lap number",
    yaxis_title="Number of samples",
    title="Samples per lap — unequal lengths require normalization",
    height=300,
    margin=dict(l=60, r=20, t=50, b=60),
)
st.plotly_chart(fig_len, use_container_width=True)
st.caption(
    "Outlier laps (pit stops, safety cars) stand out immediately as bars "
    "significantly above or below the median."
)

#Available signals 
st.markdown("## Available telemetry channels")
sig_info = pd.DataFrame([
    {"Signal": "Speed",                "Unit": "km/h",    "Range": "0-350",  "Chosen": "+"},
    {"Signal": "Throttle",             "Unit": "%",       "Range": "0-100",  "Chosen": "+"},
    {"Signal": "Brake",                "Unit": "0/1",     "Range": "0-1",    "Chosen": "+"},
    {"Signal": "RPM",                  "Unit": "rev/min", "Range": "0-15000","Chosen": "-"},
    {"Signal": "nGear",                "Unit": "—",       "Range": "1-8",    "Chosen": "-"},
    {"Signal": "DRS",                  "Unit": "state",   "Range": "0/8/12", "Chosen": "-"},
    {"Signal": "DistanceToDriverAhead","Unit": "m",       "Range": "0-∞",    "Chosen": "-"},
])

st.dataframe(sig_info, use_container_width=True, hide_index=True)