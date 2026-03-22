import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import Exercises.ex1.helper as helper
import Exercises.ex1.analysis as analysis

st.set_page_config(page_title="Discussion", layout="wide")
st.title("Discussion")

@st.cache_data
def load():
    df = helper.load_raw()
    return helper.get_driver(df)

driver_df   = load()
clean_laps  = analysis.get_clean_laps(driver_df)
summary     = analysis.build_lap_summary(driver_df)
fastest_lap = analysis.get_fastest_lap(summary)
signals     = ["Speed", "Throttle", "Brake"]

#Stint structure
st.markdown("## Stint structure")
st.markdown(
    "The three stints are clearly visible in the lap time trend. "
    "Average lap times per stint, computed directly from SessionTime, are shown in the table below. "
    "Laps 17 and 37 were excluded because lap time deteriorated strongly at the end of the tyre cycle, "
    "and the subsequent lap corresponded to the pit-stop phase. They were therefore not considered representative of normal race pace within a stint."
)

stints = [
    {"Stint": 1, "Laps": "1-16",  "From": 1,  "To": 16,  "Tyres": "Soft C3"},
    {"Stint": 2, "Laps": "19-36", "From": 19, "To": 36,  "Tyres": "Hard C1"},
    {"Stint": 3, "Laps": "39-57", "From": 39, "To": 57,  "Tyres": "Soft C3"},
]
stint_rows = []
for s in stints:
    mask = summary["Lap"].between(s["From"], s["To"])
    avg  = summary.loc[mask, "Lap time (s)"].mean()
    std  = summary.loc[mask, "Lap time (s)"].std()
    stint_rows.append({
        "Stint":        s["Stint"],
        "Laps":         s["Laps"],
        "Tyres":        s["Tyres"],
        "Avg time (s)": round(avg, 3),
        "Std (s)":      round(std, 3),
    })

stint_df = pd.DataFrame(stint_rows)
st.dataframe(stint_df, use_container_width=True, hide_index=True)

s1 = stint_df.loc[0, "Avg time (s)"]
s2 = stint_df.loc[1, "Avg time (s)"]
s3 = stint_df.loc[2, "Avg time (s)"]

st.markdown(
    f"""
    Stint 3 is the fastest ({s3} s avg) despite being on the same C3 compound as Stint 1
    ({s1} s avg). The improvement cannot be explained by tyre compound alone -
    Verstappen ran **C3 → C1 → C3** in Bahrain 2024.
    The most likely explanations are track evolution (rubber laid down over 57 laps),
    reduced fuel load, and the freshness of the final soft set.

    The standard deviation within each stint is small
    ({stint_df['Std (s)'].min():.3f}-{stint_df['Std (s)'].max():.3f} s),
    confirming highly consistent pace once tyres are up to temperature.
    """
)

#Lap time trend plot
st.markdown("## Lap time trend across the race")

fig_lt = go.Figure()

def lap_to_tyre(lap):
    if lap <= 17:   return "Soft C3",  "#e8333a"   #red for soft
    elif lap <= 38: return "Hard C1",  "#c8c8c8"   #grey for hard
    else:           return "Soft C3",  "#e8333a"   #again red for soft

all_laps_summary = summary.dropna(subset=["Lap time (s)"]).sort_values("Lap")

#continuous background line now, so, there are no gaps like in prev version...
fig_lt.add_trace(go.Scatter(
    x=all_laps_summary["Lap"],
    y=all_laps_summary["Lap time (s)"],
    mode="lines",
    line=dict(color="rgba(180,180,180,0.3)", width=1.5),
    showlegend=False,
    hoverinfo="skip",
))

#color markers per tyre compound again (on top of the background line)
for label, color, lap_range in [
    ("Soft C3 - Stint 1", "#e8333a", range(1,  18)),
    ("Hard C1 - Stint 2", "#c8c8c8", range(18, 39)),
    ("Soft C3 - Stint 3", "#e8333a", range(39, 58)),
]:
    chunk = all_laps_summary[all_laps_summary["Lap"].isin(lap_range)]
    fig_lt.add_trace(go.Scatter(
        x=chunk["Lap"],
        y=chunk["Lap time (s)"],
        mode="markers+lines",
        marker=dict(size=6, color=color),
        line=dict(color=color, width=1.5),
        name=label,
        hovertemplate="Lap %{x}: %{y:.3f} s<extra></extra>",
    ))

#also we can do pit lap markers on top, why not, since we know them for sure, and they are important reference points in the lap time trend
for pit_lap in [18, 38]:
    pit_row = all_laps_summary[all_laps_summary["Lap"] == pit_lap]
    if not pit_row.empty:
        fig_lt.add_trace(go.Scatter(
            x=pit_row["Lap"],
            y=pit_row["Lap time (s)"],
            mode="markers",
            marker=dict(size=12, color="yellow", symbol="diamond", #can be any shape/symbol
                        line=dict(color="black", width=1)),
            name=f"Pit stop (Lap {pit_lap})",
            hovertemplate=f"Pit lap {pit_lap}: %{{y:.3f}} s<extra></extra>",
        ))

#of course pit stop vertical lines
for pit_lap in [18, 38]:
    fig_lt.add_vline(
        x=pit_lap,
        line=dict(color="rgba(255,255,0,0.4)", dash="dash", width=1),
    )

fig_lt.update_layout(
    xaxis=dict(
        title="Lap number",
        range=[1, 57],
        dtick=5,
        tick0=1,
    ),
    yaxis_title="Lap time (s)",
    height=400,
    margin=dict(l=60, r=20, t=40, b=60),
    legend=dict(orientation="h", y=1.02),
    hovermode="x unified",
)
st.plotly_chart(fig_lt, use_container_width=True)
st.caption(
    "Red = Soft C3, grey = Hard C1. "
    "Yellow diamonds = pit stop laps (18 and 38). "
    "The x-axis runs continuously from lap 1 to 57 - "
    "transition laps 170->18 and 37->38 are included."
)

#Consistency table 
st.markdown("## How consistent is Verstappen?")
st.markdown(
    "CV < 5% means the driver repeats that input reliably lap after lap. "
    "Higher CV reveals where variability is introduced deliberately or by external factors."
)
cv_df = analysis.consistency_score(driver_df, signals)
st.dataframe(cv_df, use_container_width=True, hide_index=True)

#Brake analysis
st.markdown("## Braking consistency across the race")
st.markdown(
    """
    We check whether braking zones get longer as tyres degrade -
    less grip would force earlier, longer brake applications.
    One representative lap is sampled from each part of each stint:
    """
)

sample_laps = [5, 17, 22, 37, 42, 54]
brake_rows  = []
for lap in sample_laps:
    ldf = helper.get_lap(driver_df, lap).copy()
    ldf["time_in_lap_s"] = ldf[helper.COL_TIME].dt.total_seconds()
    ldf = ldf.dropna(subset=["time_in_lap_s", "Speed"]).sort_values("time_in_lap_s")
    b   = analysis.get_brake_section_summary(ldf)
    if not b.empty:
        stint_num = 1 if lap <= 16 else (2 if lap <= 36 else 3)
        brake_rows.append({
            "Lap":                   lap,
            "Stint":                 stint_num,
            "Braking zones":         len(b),
            "Total brake time (s)":  round(b["Duration (s)"].sum(), 2),
            "Avg speed drop (km/h)": round(b["Speed drop (km/h)"].mean(), 1),
        })

brake_trend = pd.DataFrame(brake_rows)
st.dataframe(brake_trend, use_container_width=True, hide_index=True)
st.caption(
    "Total brake time stays within 18.6-20.0 s across all three stints - "
    "no clear degradation trend is visible in the time domain. "
    "This either means tyre wear did not significantly affect braking at Bahrain, "
    "or that our lap sampling (one lap per stint segment) is too coarse to detect it. "
    "Probably a corner-by-corner breakdown using GPS alignment would be needed to say more."
)

#Fastest lap breakdown 
st.markdown(f"## Fastest lap breakdown - Lap {fastest_lap}")
fl_df = helper.get_lap(driver_df, fastest_lap).copy()
fl_df["time_in_lap_s"] = fl_df[helper.COL_TIME].dt.total_seconds()
fl_df = fl_df.dropna(subset=["time_in_lap_s", "Speed"]).sort_values("time_in_lap_s")
fl_brake = analysis.get_brake_section_summary(fl_df)

c1, c2, c3 = st.columns(3)
c1.metric("Lap time",        f"{summary.loc[summary['Lap']==fastest_lap, 'Lap time (s)'].values[0]:.3f} s")
c2.metric("Top speed",       f"{fl_df['Speed'].max():.1f} km/h")
c3.metric("Braking zones",   len(fl_brake))

if not fl_brake.empty:
    st.dataframe(fl_brake, use_container_width=True, hide_index=True)
    st.caption(
        "Each row is one braking zone. "
        "Speed drop = entry speed minus minimum speed reached. "
        "The longest duration zones correspond to the heaviest braking points on the circuit."
    )

st.divider()
st.caption("DSP Exercise 1 * FH Joanneum * 2026")