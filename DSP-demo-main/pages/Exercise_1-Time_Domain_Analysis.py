import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Exercise 1: Time Domain Analysis", layout="wide")


DATA_PATH = "Bahrain_time_series.csv"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Convert time columns
    if "Time" in df.columns:
        df["Time"] = pd.to_timedelta(df["Time"], errors="coerce")

    if "SessionTime" in df.columns:
        df["SessionTime"] = pd.to_timedelta(df["SessionTime"], errors="coerce")

    # Convert numeric columns
    for col in ["LapNumber", "Speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert Brake to binary 0/1
    if "Brake" in df.columns:
        df["Brake_active"] = (
            df["Brake"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )
    else:
        df["Brake_active"] = 0

    return df


def build_ver_data(df):
    ver = df[df["Driver"] == "VER"].copy()

    if ver.empty:
        return pd.DataFrame()

    ver = ver.dropna(subset=["LapNumber"])
    ver["LapNumber"] = ver["LapNumber"].astype(int)

    # Time within lap in seconds
    if "Time" in ver.columns and ver["Time"].notna().any():
        ver["time_in_lap_s"] = ver["Time"].dt.total_seconds()
    else:
        ver["time_in_lap_s"] = ver.groupby("LapNumber").cumcount().astype(float)

    return ver


def build_summary(ver):
    summary = ver.groupby("LapNumber").agg(
        avg_speed=("Speed", "mean"),
        max_speed=("Speed", "max")
    ).reset_index()

    if "SessionTime" in ver.columns and ver["SessionTime"].notna().any():
        lap_times = ver.groupby("LapNumber")["SessionTime"].agg(["min", "max"]).reset_index()
        lap_times["lap_time_s"] = (lap_times["max"] - lap_times["min"]).dt.total_seconds()
        summary = summary.merge(
            lap_times[["LapNumber", "lap_time_s"]],
            on="LapNumber",
            how="left"
        )
    else:
        summary["lap_time_s"] = np.nan

    summary = summary.rename(columns={
        "LapNumber": "Lap",
        "lap_time_s": "Lap time (s)",
        "avg_speed": "Average speed",
        "max_speed": "Max speed"
    })

    summary = summary[["Lap", "Lap time (s)", "Average speed", "Max speed"]]
    summary["Lap time (s)"] = summary["Lap time (s)"].round(4)
    summary["Average speed"] = summary["Average speed"].round(4)
    summary["Max speed"] = summary["Max speed"].round(4)

    return summary.sort_values("Lap").reset_index(drop=True)


def get_fastest_lap(summary):
    valid = summary.dropna(subset=["Lap time (s)"]).copy()
    if valid.empty:
        return None
    return int(valid.loc[valid["Lap time (s)"].idxmin(), "Lap"])


def get_max_speed_lap(summary):
    valid = summary.dropna(subset=["Max speed"]).copy()
    if valid.empty:
        return None
    return int(valid.loc[valid["Max speed"].idxmax(), "Lap"])


def get_lap_and_previous(summary, lap):
    current = summary[summary["Lap"] == lap].copy()
    previous = summary[summary["Lap"] == lap - 1].copy()
    return current, previous


def get_lap_df(ver, lap_number):
    lap_df = ver[ver["LapNumber"] == lap_number].copy()
    lap_df = lap_df.dropna(subset=["time_in_lap_s", "Speed"])
    lap_df = lap_df.sort_values("time_in_lap_s").reset_index(drop=True)
    return lap_df


def get_brake_segments(lap_df):
    x = lap_df["time_in_lap_s"].to_numpy()
    b = lap_df["Brake_active"].to_numpy()

    segments = []
    in_segment = False
    start = None

    for i in range(len(lap_df)):
        if b[i] == 1 and not in_segment:
            start = x[i]
            in_segment = True
        elif b[i] == 0 and in_segment:
            end = x[i]
            segments.append((start, end))
            in_segment = False

    if in_segment and len(x) > 0:
        segments.append((start, x[-1]))

    return segments


def zero_speed_duration(lap_df):
    if lap_df.empty or len(lap_df) < 2:
        return 0.0

    lap_df = lap_df.sort_values("time_in_lap_s").reset_index(drop=True).copy()
    lap_df["dt_next"] = lap_df["time_in_lap_s"].shift(-1) - lap_df["time_in_lap_s"]
    lap_df["dt_next"] = lap_df["dt_next"].fillna(0)

    duration = lap_df.loc[lap_df["Speed"] == 0, "dt_next"].sum()
    return float(duration)


def plot_lap_speed(ver, lap_number, title):
    lap_df = get_lap_df(ver, lap_number)

    fig, ax = plt.subplots(figsize=(12, 5))

    for start, end in get_brake_segments(lap_df):
        ax.axvspan(start, end, alpha=0.15, color="red")

    ax.plot(
        lap_df["time_in_lap_s"].to_numpy(),
        lap_df["Speed"].to_numpy(),
        linewidth=2
    )

    ax.set_title(title)
    ax.set_xlabel("Time within lap (s)")
    ax.set_ylabel("Speed")
    ax.grid(True, alpha=0.3)

    return fig, lap_df


def get_brake_section_summary(lap_df):
    lap_df = lap_df.sort_values("time_in_lap_s").reset_index(drop=True).copy()

    if lap_df.empty:
        return pd.DataFrame()

    x = lap_df["time_in_lap_s"].to_numpy()
    b = lap_df["Brake_active"].to_numpy()
    v = lap_df["Speed"].to_numpy()

    segments = []
    in_segment = False
    start_idx = None

    for i in range(len(lap_df)):
        if b[i] == 1 and not in_segment:
            start_idx = i
            in_segment = True
        elif b[i] == 0 and in_segment:
            end_idx = i - 1
            segments.append((start_idx, end_idx))
            in_segment = False

    if in_segment:
        segments.append((start_idx, len(lap_df) - 1))

    rows = []

    for seg_no, (i0, i1) in enumerate(segments, start=1):
        start_t = float(x[i0])

        if i1 < len(lap_df) - 1:
            end_t = float(x[i1 + 1])
        else:
            end_t = float(x[i1])

        duration = max(0.0, end_t - start_t)

        rows.append({
            "Brake section": seg_no,
            "Brake duration (s)": round(duration, 4),
            "Speed at brake start": round(float(v[i0]), 4),
            "Speed at brake end": round(float(v[i1]), 4),
        })

    return pd.DataFrame(rows)


st.title("Exercise 1: Time Domain Analysis")
st.subheader("Max Verstappen (VER) Lap Summary")

df = load_data(DATA_PATH)

if "Driver" not in df.columns:
    st.error("Column 'Driver' not found.")
    st.stop()

ver = build_ver_data(df)

if ver.empty:
    st.error("No data found for driver VER.")
    st.stop()

summary = build_summary(ver)

st.markdown("### Lap table")
st.table(summary.reset_index(drop=True))

fastest_lap = get_fastest_lap(summary)
max_speed_lap = get_max_speed_lap(summary)

st.markdown("### Key lap analysis")

if fastest_lap is not None:
    current, previous = get_lap_and_previous(summary, fastest_lap)
    st.markdown(f"**Fastest lap:** Lap {fastest_lap}")
    st.table(current.reset_index(drop=True))

    if not previous.empty:
        st.markdown("**Previous lap:**")
        st.table(previous.reset_index(drop=True))

        prev_lap_df = get_lap_df(ver, fastest_lap - 1)
        zero_duration = zero_speed_duration(prev_lap_df)
        st.write(f"Time at speed 0 in the previous lap of the fastest lap: **{zero_duration:.4f} s**")

# if max_speed_lap is not None:
#     current, previous = get_lap_and_previous(summary, max_speed_lap)
#     st.markdown(f"**Lap with highest max speed:** Lap {max_speed_lap}")
#     st.table(current.reset_index(drop=True))

#     if not previous.empty:
#         st.markdown("**Previous lap:**")
#         st.table(previous.reset_index(drop=True))

# --- Stint analysis ---

def mean_lap_time(summary_df, lap_start, lap_end):
    laps = summary_df[(summary_df["Lap"] >= lap_start) & (summary_df["Lap"] <= lap_end)].copy()
    return float(laps["Lap time (s)"].mean())

def zero_speed_duration_for_lap(ver_df, lap_number):
    lap_df = get_lap_df(ver_df, lap_number)
    return zero_speed_duration(lap_df)

avg_1_16 = mean_lap_time(summary, 1, 16)
zero_18 = zero_speed_duration_for_lap(ver, 18)
avg_19_36 = mean_lap_time(summary, 19, 36)
avg_39_57 = mean_lap_time(summary, 39, 57)

st.markdown("### Stint interpretation")

st.write(
    "One can see that the lap times from Lap 1 to Lap 16 are relatively constant, "
    "with an average lap time of 96.8459 s. Lap 17 is already clearly worse, "
    "which suggests that the tyres start to degrade. In the following lap, Lap 18, "
    "the lap time is extremely high. If one looks at the standing time in this lap, "
    "that is the time with speed = 0, it is 3.0800 s. This strongly suggests that "
    "Verstappen made a pit stop in this lap. After that, the lap times from Lap 19 "
    "to Lap 36 are again relatively constant, with an average lap time of 95.4291 s. "
    "In Lap 37, tyre performance starts to drop again, and Lap 38 is another pit stop lap. "
    "The following lap is also his fastest lap, which is consistent with the use of new tyres. "
    "The average lap time from Lap 39 to Lap 57 is 94.5162 s. From the lap times alone, "
    "one might initially assume that Verstappen started on a harder compound and later switched "
    "to a softer one. However, this is not correct. In Bahrain 2024, Verstappen started on the "
    "soft tyre (C3), then switched to the hard tyre (C1), and for the final stint Red Bull used "
    "a fresh set of C3 soft tyres rather than another hard set. Therefore, the improvement in lap "
    "times in the final part of the race can more plausibly be linked to the pit stop, fresher tyres, "
    "and possible track evolution."
)

st.markdown("### Lap 1 speed plot")
fig1, lap1_df = plot_lap_speed(ver, 1, "Lap 1: Speed profile with brake-active zones")
st.write(f"Number of plotted points in Lap 1: {len(lap1_df)}")
st.pyplot(fig1)

if fastest_lap is not None:
    st.markdown("### Fastest lap speed plot")
    fig2, fastest_df = plot_lap_speed(
        ver,
        fastest_lap,
        f"Lap {fastest_lap}: Speed profile with brake-active zones (fastest lap)"
    )
    st.write(f"Number of plotted points in fastest lap: {len(fastest_df)}")
    st.pyplot(fig2)

if max_speed_lap is not None:
    st.markdown("### Max speed lap plot")
    fig3, maxspeed_df = plot_lap_speed(
        ver,
        max_speed_lap,
        f"Lap {max_speed_lap}: Speed profile with brake-active zones (highest max speed)"
    )
    st.write(f"Number of plotted points in max speed lap: {len(maxspeed_df)}")
    st.pyplot(fig3)

st.markdown("### Brake-section comparison: fastest lap vs. max-speed lap")
st.write(
    "The tables below compare the braking sections of the fastest lap and the lap with the highest maximum speed. "
    "For each braking section, the brake duration as well as the speed at brake start and brake end are shown."
)

if fastest_lap is not None and max_speed_lap is not None:
    fastest_df = get_lap_df(ver, fastest_lap)
    maxspeed_df = get_lap_df(ver, max_speed_lap)

    fastest_brake_sections = get_brake_section_summary(fastest_df)
    maxspeed_brake_sections = get_brake_section_summary(maxspeed_df)

    st.markdown(f"**Fastest lap: Lap {fastest_lap}**")
    if fastest_brake_sections.empty:
        st.write("No brake sections found.")
    else:
        st.table(fastest_brake_sections.reset_index(drop=True))

    st.markdown(f"**Lap with highest max speed: Lap {max_speed_lap}**")
    if maxspeed_brake_sections.empty:
        st.write("No brake sections found.")
    else:
        st.table(maxspeed_brake_sections.reset_index(drop=True))
else:
    st.write("Fastest lap or max-speed lap could not be determined.")