import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import helper
import analysis

st.set_page_config(page_title="Discussion", page_icon="💬", layout="wide")
st.title("💬 Discussion")

@st.cache_data
def load():
    df = helper.load_raw()
    return helper.get_driver(df)

driver_df   = load()
clean_laps  = analysis.get_clean_laps(driver_df)
summary     = analysis.build_lap_summary(driver_df)
fastest_lap = analysis.get_fastest_lap(summary)
signals     = ["Speed", "Throttle", "Brake"]

# ── Stint structure ───────────────────────────────────────────────────────────
st.markdown("## Stint structure")
st.markdown(
    "The three stints are clearly visible in the lap time trend. "
    "Average lap times per stint, computed directly from SessionTime:"
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
    ({s1} s avg). The improvement cannot be explained by tyre compound alone —
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
for s in stints:
    mask  = summary["Lap"].between(s["From"], s["To"])
    chunk = summary[mask]
    fig_lt.add_trace(go.Scatter(
        x=chunk["Lap"],
        y=chunk["Lap time (s)"],
        mode="lines+markers",
        name=f"Stint {s['Stint']} ({s['Tyres']})",
        marker=dict(size=5),
    ))

#Pit stop markers
for pit_lap in [18, 38]:
    fig_lt.add_vline(
        x=pit_lap,
        line=dict(color="red", dash="dash", width=1.5),
        annotation_text=f"Pit lap {pit_lap}",
        annotation_position="top",
    )

fig_lt.update_layout(
    xaxis_title="Lap number",
    yaxis_title="Lap time (s)",
    height=380,
    margin=dict(l=60, r=20, t=40, b=60),
    legend=dict(orientation="h", y=1.02),
    hovermode="x unified",
)
st.plotly_chart(fig_lt, use_container_width=True)
st.caption(
    "Each stint is colored separately. Red dashed lines mark pit stops on laps 18 and 38, "
    "detected from sustained Speed = 0 in the telemetry."
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
    Do braking zones get longer as tyres degrade?
    Longer brake duration at the same corner = less grip = harder stops needed.
    We sample one representative lap from each part of each stint:
    """
)

sample_laps = [5, 14, 22, 34, 42, 54]
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
    "An increase in total brake time within a stint = tyre degradation. "
    "A reset to early-stint values after a pit stop = fresh tyre effect. "
    "This is the most direct time-domain evidence of tyre wear in this dataset."
)

#Fastest lap breakdown 
st.markdown(f"## Fastest lap breakdown — Lap {fastest_lap}")
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

#Honest assessment
st.markdown("## What worked, what didn't, what's next")
st.dataframe(pd.DataFrame([
    {"What":    "Lap normalization (0-100%)",
     "Verdict": "✅ Works",
     "Limit":   "Linear interp — can misalign if a lap has an unusually slow sector"},
    {"What":    "Pit stop detection via Speed=0",
     "Verdict": "✅ Works",
     "Limit":   "Assumes pitlane speed limiter brings car to full stop — holds for Bahrain"},
    {"What":    "TrackStatus outlier filtering",
     "Verdict": "✅ Works",
     "Limit":   "Excludes entire laps — a yellow in sector 3 also removes sectors 1 and 2"},
    {"What":    "CV consistency metric",
     "Verdict": "✅ Informative",
     "Limit":   "Doesn't separate within-lap vs lap-to-lap variability"},
    {"What":    "Brake zone detection (threshold)",
     "Verdict": "⚠️ Basic",
     "Limit":   "No smoothing — sensor noise can create false positives"},
    {"What":    "Distance-based lap alignment",
     "Verdict": "❌ Not done",
     "Limit":   "Would need GPS distance channel — time-based alignment is an approximation"},
    {"What":    "FFT on Speed signal",
     "Verdict": "❌ Not done",
     "Limit":   "Would reveal lap-periodic structure — planned for Exercise 2"},
]), use_container_width=True, hide_index=True)

st.divider()
st.caption("DSP Exercise 1 · FH Joanneum · 2026")