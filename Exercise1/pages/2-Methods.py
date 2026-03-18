import streamlit as st
import sys, plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent))

import helper 
import analysis

st.set_page_config(page_title="Methods", layout="wide")
st.title("Methods")

@st.cache_data
def load():
    df = helper.load_raw()
    return helper.get_driver(df)

driver_df = load()

#Variable selection
st.markdown("## Why these three variables?")
st.markdown(
    "Speed, Throttle, and Brake describe the complete driver input cycle. "
    "We show the correlation matrix below that confirms their complementarity: they are not redundant, but rather "
    "interdependent signals that together carry distinct information:"
)

corr = (
    driver_df[["Speed", "Throttle", "Brake_active"]]
    .dropna()
    .corr()
    .round(3)
)
st.dataframe(corr, use_container_width=True)
st.caption(
    "Brake_active is anti-correlated with both Speed and Throttle — "
    "it activates exactly when the other two drop. "
    "This means each signal adds non-redundant information."
)

#Lap time calculation
st.markdown("## How lap times are computed")
st.markdown(
    """
    Lap time is derived from the `SessionTime` column — cumulative time since
    session start. For each lap:

    `lap_time = SessionTime.max() - SessionTime.min()`

    This is more reliable than using the `Time` delta column directly,
    which resets at lap boundaries and can contain small gaps.
    """
)
summary = analysis.build_lap_summary(driver_df)
fig_lt = go.Figure(go.Histogram(
    x=summary["Lap time (s)"].dropna(),
    xbins=dict(
        start=summary["Lap time (s)"].min() - 0.5,
        end=summary["Lap time (s)"].max() + 1.5,
        size=1,                          #1 second per bin
    ),
    marker_color="cornflowerblue",
))
fig_lt.update_layout(
    xaxis_title="Lap time (s)",
    yaxis_title="Count",
    title="Distribution of lap times - pit stop laps marked in red",
    height=300,
    margin=dict(l=60, r=20, t=50, b=60),
)
pit_lap_times = {}
for pit_lap in [18, 38]:
    row = summary[summary["Lap"] == pit_lap]["Lap time (s)"]
    if not row.empty:
        pit_lap_times[pit_lap] = row.values[0]

for pit_lap, pit_time in pit_lap_times.items():
    fig_lt.add_vline(
        x=pit_time,
        line=dict(color="tomato", dash="dash", width=1.5),
        annotation_text=f"Lap {pit_lap} ({pit_time:.1f}s)",
        annotation_position="top right" if pit_lap == 18 else "top left",
        annotation_font=dict(color="tomato", size=11),
    )
st.plotly_chart(fig_lt, use_container_width=True)
# st.caption(
#     "Most racing laps cluster between 94-97 s. "
#     "The bar at ~99-100 s is a slightly slower lap, likely due to traffic or a yellow flag. "
#     "The isolated bar at ~117 s is a confirmed pit stop lap — "
#     "identified by sustained Speed = 0 in the zero-speed analysis below. "
#     "The second pit stop lap may fall within the main cluster depending on pitlane duration."
# )
st.caption(
    f"Main cluster: 94-97 s racing laps. "
    f"Lap 18: {pit_lap_times.get(18, '?'):.1f} s — "
    f"Lap 38: {pit_lap_times.get(38, '?'):.1f} s. "
    "Both are marked with red dashed lines. "
    "If one falls within the main cluster, it means that pit stop was unusually fast "
    "and the car was moving through the pitlane for most of the lap."
)

#Pit stop detection
st.markdown("## Pit stop detection via Speed = 0")
st.markdown(
    """
    We detect pit stop laps by summing the time intervals where Speed = 0.
    A racing lap has zero such time; a pit lap accumulates 3+ seconds
    stationary in the pitlane. This is a purely time-domain method —
    no labels or external data needed, just the signal amplitude hitting zero.
    """
)
st.markdown(
    """
    Note: the two pit stop laps appear at different positions in the histogram
    because pit stop duration varied — one stop was significantly longer than
    the other. This is confirmed in the zero-speed analysis below.
    """
)

pit_rows = []
for lap in [16, 17, 18, 19, 36, 37, 38, 39]:
    ldf = helper.get_lap(driver_df, lap).copy()
    ldf["time_in_lap_s"] = ldf[helper.COL_TIME].dt.total_seconds()
    t0 = analysis.zero_speed_duration(ldf)
    pit_rows.append({
        "Lap":                lap,
        "Time at Speed=0 (s)":round(t0, 3),
        "Verdict":            "Pit stop" if t0 > 1.0 else "Racing lap",
    })
st.dataframe(pd.DataFrame(pit_rows), use_container_width=True, hide_index=True)
st.caption("Laps 18 and 38 are the only laps with sustained zero-speed periods.")


#Outlier detection 
st.markdown("## Outlier lap detection via TrackStatus")
st.markdown(
    """
    Instead of guessing with a Z-score, we use the **TrackStatus** flag logged
    by the FIA timing system. Any lap that contains a non-green status is
    excluded from statistical summaries.
    """
)

status_df = analysis.flag_outlier_laps(driver_df)
n_flagged = status_df["Flagged"].sum()
n_clean   = (~status_df["Flagged"]).sum()

c1, c2, c3 = st.columns(3)
c1.metric("Total laps", len(status_df))
c2.metric("Clean laps (green flag)", int(n_clean))
c3.metric("Flagged laps", int(n_flagged))

st.dataframe(
    status_df.style.apply(
        lambda r: ["background-color: rgba(255,80,80,0.15)"] * len(r)
                  if r["Flagged"] else [""] * len(r),
        axis=1,
    ),
    use_container_width=True,
    hide_index=True,
)
st.caption("Red rows are excluded from all multi-lap statistics and overlay plots.")


#Lap normalization
st.markdown("## Handling variable-length laps")
st.markdown(
    """
    The lap time histogram above shows that even clean racing laps vary between
    roughly 94-97 s, while the confirmed pit stop lap reaches ~117 s. Since the data is sampled
    at a fixed rate, a longer lap = more samples. Trying to average a 350-sample
    lap with a 420-sample lap directly would misalign every corner feature.
    We solve this by resampling every lap onto a 500-point 0-100% grid
    using linear interpolation. The before/after comparison below shows
    the alignment improvement:
    """
)

demo        = analysis.normalization_demo(driver_df, signal="Speed")
raw, norm   = demo["raw"], demo["normalized"]
sample_laps = list(raw.keys())
colors      = ["cornflowerblue", "coral", "mediumseagreen"]

fig_norm = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Raw — different sample counts",
        "Normalized — 0-100% lap completion",
    ),
)
for i, lap in enumerate(sample_laps):
    fig_norm.add_trace(go.Scatter(
        x=raw[lap]["x"], y=raw[lap]["y"],
        mode="lines", line=dict(color=colors[i], width=1.5),
        name=f"Lap {lap}",
    ), row=1, col=1)
    fig_norm.add_trace(go.Scatter(
        x=norm[lap]["x"], y=norm[lap]["y"],
        mode="lines", line=dict(color=colors[i], width=1.5),
        name=f"Lap {lap}", showlegend=False,
    ), row=1, col=2)

fig_norm.update_layout(height=380, margin=dict(l=60, r=20, t=60, b=60))
fig_norm.update_xaxes(title_text="Sample index",       row=1, col=1)
fig_norm.update_xaxes(title_text="Lap completion (%)", row=1, col=2)
fig_norm.update_yaxes(title_text="Speed (km/h)",       row=1, col=1)
st.plotly_chart(fig_norm, use_container_width=True)
st.caption(
    "Left: three laps end at different sample counts — they cannot be averaged directly. "
    "Right: after interpolation to 500 points, corner features align at the same x position."
)


#Consistency metric 
st.markdown(
    """
    ## Consistency metric: Coefficient of Variation (CV)

    Standard deviation alone is hard to compare across signals with very different
    magnitudes (Speed in hundreds vs Brake in 0-1).  
    We use **CV = sigma / mu x 100 (%)** — a normalized measure of spread:

    |   CV   |                      Interpretation                           |
    |--------|---------------------------------------------------------------|
    | < 5 %  | Highly consistent — driver repeats this input reliably        |
    | 5-15 % | Moderate variation — expected for complex inputs like braking |
    | > 15 % | High variation — anomalous laps or deliberate strategy change |
    """
)