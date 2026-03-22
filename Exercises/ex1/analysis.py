import numpy as np
import pandas as pd
import plotly.graph_objects as go 
from Exercises.ex1.helper import (
    get_lap, list_laps,
    lap_summary, COL_LAP, COL_TIME, COL_SESSION
)

#Sampling rate
def compute_sampling_stats(driver_df: pd.DataFrame) -> dict:
    """
    Compute real sampling rate from the Time column (delta between rows). Returns mean fs, std, min, max in Hz.
    """
    #driver_df["Time"] = pd.to_timedelta(driver_df["Time"])
    deltas = driver_df[COL_TIME].diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0] #drop resets at lap boundaries
    fs_series = 1.0 / deltas
    return {
        "mean_hz":   round(fs_series.mean(), 2),
        "std_hz":    round(fs_series.std(), 2),
        "min_hz":    round(fs_series.min(), 2),
        "max_hz":    round(fs_series.max(), 2),
        "median_dt": round(deltas.median() * 1000, 2),   #ms
        "n_samples": len(driver_df),
        "n_laps":    driver_df[COL_LAP].nunique(),
    }


#Lap lengths 
def compute_lap_lengths(driver_df: pd.DataFrame) -> pd.DataFrame:
    """Number of samples per lap - shows why normalization is needed."""
    return (
        driver_df.groupby(COL_LAP)
        .size()
        .reset_index(name="n_samples")
        .rename(columns={COL_LAP: "Lap"})
    )


#Outlier lap detection via TrackStatus
def flag_outlier_laps(driver_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with every lap, its dominant TrackStatus, and whether it's flagged. Status '1' or '2' (green/yellow) -> clean lap.
    """
    track_status = pd.to_numeric(driver_df["TrackStatus"], errors="coerce")
    df = driver_df.assign(_TrackStatus=track_status)

    status_per_lap = (
        df.groupby(COL_LAP)["_TrackStatus"]
        .agg(
            TrackStatus=lambda s: (
                str(int(s.dropna().astype(int).value_counts().idxmax()))
                if s.notna().any()
                else np.nan
            ),
            Flagged=lambda s: (~s.isin([1, 1])).any(),
        ).reset_index().rename(columns={COL_LAP: "Lap"})
    )
    status_map = { "1": "Green", "2": "Yellow", "4": "Safety Car", "5": "Red Flag", "6": "VSC ending" }
    status_per_lap["Status Label"] = ( status_per_lap["TrackStatus"].map(status_map).fillna("Unknown") )
    return status_per_lap


def get_clean_laps(driver_df: pd.DataFrame) -> list[int]:
    df = flag_outlier_laps(driver_df)
    return df.loc[~df["Flagged"], "Lap"].tolist()


#Normalization demo 
def normalization_demo(driver_df: pd.DataFrame, signal: str = "Speed", sample_laps: list = None) -> dict:
    """
    For the Methods page: return raw + normalized versions of 3 sample laps to visualize why resampling is necessary.
    """
    laps = list_laps(driver_df)
    if sample_laps is None:
        #we pick 3 spread-out laps
        sample_laps = [laps[5], laps[17], laps[-5]]

    raw, normalized = {}, {}
    for lap in sample_laps:
        lap_df = get_lap(driver_df, lap)
        raw[lap] = {
            "x": np.arange(len(lap_df)),
            "y": lap_df[signal].values,
        }
        x_norm = np.linspace(0, 100, 500)
        y_norm = np.interp(x_norm, np.linspace(0, 100, len(lap_df)), lap_df[signal].values)
        normalized[lap] = {"x": x_norm, "y": y_norm}

    return {"raw": raw, "normalized": normalized, "signal": signal}


#Lap matrix + statistics
def full_lap_statistics(driver_df: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """
    Per-lap statistics that are actually meaningful:
    - Speed: max (top speed on straight), min (slowest corner)
    - Throttle: mean (% of lap at throttle)
    - Brake: sum of samples braking / total samples (% of lap braking)
    - RPM: max
    """
    frames = []
    for sig in signals:
        if sig not in driver_df.columns:
            continue
        grp = driver_df.groupby(COL_LAP)[sig]

        if sig == "Speed":
            s = pd.DataFrame({
                "Lap":         grp.max().index,
                "Speed_max":   grp.max().values.round(1),   # top speed
                "Speed_min":   grp.min().values.round(1),   # min corner speed
                "Speed_mean":  grp.mean().values.round(1),  # keep for reference
                "Speed_std":   grp.std().values.round(2),   # within-lap (large by nature)
            })
        elif sig == "Throttle":
            s = pd.DataFrame({
                "Lap":            grp.mean().index,
                "Throttle_mean":  grp.mean().values.round(2),  # avg throttle %
                "Throttle_full":  # % of samples at full throttle (>98%)
                    (driver_df.groupby(COL_LAP)[sig].apply(lambda x: (x > 98).mean() * 100).values.round(2)),
            })
        elif sig == "Brake":
            s = pd.DataFrame({
                "Lap":            grp.mean().index,
                "Brake_pct":      # % of lap spent braking
                    (driver_df.groupby(COL_LAP)[sig].apply(lambda x: (x > 0.1).mean() * 100).values.round(2)),
                "Brake_max":      grp.max().astype(float).round(3).values
            })
        else:
            s = pd.DataFrame({
                "Lap":           grp.mean().index,
                f"{sig}_mean":   grp.mean().values.round(2),
                f"{sig}_std":    grp.std().values.round(3),
            })

        frames.append(s.set_index("Lap"))

    return pd.concat(frames, axis=1).reset_index()


#Braking / event detection 
def detect_braking_events(lap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds braking zones in a single lap using the Brake column. Returns a DataFrame with start_idx, end_idx, duration_samples, min_speed_in_zone, max_brake.
    """
    brake = lap_df["Brake"].fillna(0).values
    #binarize: anything > 0.1 counts as braking
    braking = (brake > 0.1).astype(int)
    diff    = np.diff(braking, prepend=0, append=0)
    starts  = np.where(diff == 1)[0]
    ends    = np.where(diff == -1)[0]

    events = []
    for s, e in zip(starts, ends):
        zone = lap_df.iloc[s:e]
        if len(zone) < 3:   #we ignore any tiny glitches if any
            continue
        events.append({
            "start_idx":    s,
            "end_idx":      e,
            "duration_pts": e - s,
            "min_speed":    round(zone["Speed"].min(), 1) if "Speed" in zone else None,
            "max_brake":    round(float(zone["Brake"].max()), 3) if "Brake" in zone else None, #must be float for some reason to avoid numpy int error when max is 0
        })
    return pd.DataFrame(events)


#Consistency score 
def consistency_score(driver_df: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """
    Summary table: mean CV per signal across clean laps. Lower CV = more consistent driving.
    """
    clean = get_clean_laps(driver_df)
    clean_df = driver_df[driver_df[COL_LAP].isin(clean)]
    rows = []
    for sig in signals:
        if sig not in clean_df.columns:
            continue
        s = lap_summary(clean_df, sig)
        cv_mean = (s["std"] / s["mean"].abs() * 100).mean()
        rows.append({
            "Signal": sig,
            "Mean across laps": round(s["mean"].mean(), 2),
            "Avg std": round(s["std"].mean(), 3),
            "CV (%)": round(cv_mean, 2),
            "Consistency": "High" if cv_mean < 5 else "Medium" if cv_mean < 15 else "Low",
        })
    return pd.DataFrame(rows)


def build_lap_summary(driver_df: pd.DataFrame) -> pd.DataFrame:
    summary = driver_df.groupby(COL_LAP).agg(
        avg_speed=("Speed", "mean"),
        max_speed=("Speed", "max"),
    ).reset_index()

    if COL_SESSION in driver_df.columns and driver_df[COL_SESSION].notna().any():
        lap_times = (
            driver_df.groupby(COL_LAP)[COL_SESSION] 
            .agg(["min", "max"])
            .reset_index()
        )
        lap_times["lap_time_s"] = (
            lap_times["max"] - lap_times["min"]
        ).dt.total_seconds()
        summary = summary.merge(
            lap_times[[COL_LAP, "lap_time_s"]], on=COL_LAP, how="left"
        )
    else:
        summary["lap_time_s"] = np.nan

    return summary.rename(columns={
        COL_LAP:      "Lap",
        "lap_time_s": "Lap time (s)",
        "avg_speed":  "Avg speed (km/h)",
        "max_speed":  "Max speed (km/h)",
    }).round(3)


def get_fastest_lap(summary: pd.DataFrame) -> int | None:
    valid = summary.dropna(subset=["Lap time (s)"])
    if valid.empty:
        return None
    return int(valid.loc[valid["Lap time (s)"].idxmin(), "Lap"])


def get_brake_section_summary(lap_df: pd.DataFrame) -> pd.DataFrame:
    """Detailed per-braking-zone table."""
    lap_df = lap_df.sort_values("time_in_lap_s").reset_index(drop=True)
    x = lap_df["time_in_lap_s"].to_numpy()
    b = lap_df["Brake_active"].to_numpy()
    v = lap_df["Speed"].to_numpy()

    segments, in_seg, start_idx = [], False, None
    for i in range(len(lap_df)):
        if b[i] == 1 and not in_seg:
            start_idx, in_seg = i, True
        elif b[i] == 0 and in_seg:
            segments.append((start_idx, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start_idx, len(lap_df) - 1))

    rows = []
    for n, (i0, i1) in enumerate(segments, 1):
        end_t = float(x[i1 + 1]) if i1 < len(lap_df) - 1 else float(x[i1])
        rows.append({
            "Zone":                  n,
            "Duration (s)":          round(max(0.0, end_t - float(x[i0])), 3),
            "Speed at brake start":  round(float(v[i0]), 1),
            "Speed at brake end":    round(float(v[i1]), 1),
            "Speed drop (km/h)":     round(float(v[i0]) - float(v[i1]), 1),
        })
    return pd.DataFrame(rows)


def zero_speed_duration(lap_df: pd.DataFrame) -> float:
    """Detects pit stops via time spent at speed=0."""
    if lap_df.empty or len(lap_df) < 2:
        return 0.0
    lap_df = lap_df.sort_values("time_in_lap_s").reset_index(drop=True).copy()
    lap_df["dt"] = lap_df["time_in_lap_s"].shift(-1) - lap_df["time_in_lap_s"]
    return float(lap_df.loc[lap_df["Speed"] == 0, "dt"].fillna(0).sum())


def add_brake_zones(fig, lap_df, row=1):
    """Adds red shading to Plotly figure."""
    x    = lap_df["time_in_lap_s"].values
    b    = lap_df["Brake_active"].values
    in_seg, start = False, None

    for i in range(len(x)):
        if b[i] == 1 and not in_seg:
            start, in_seg = x[i], True
        elif b[i] == 0 and in_seg:
            fig.add_vrect(
                x0=start, x1=x[i],
                fillcolor="red", opacity=0.12,
                line_width=0, row=row, col=1,
            )
            in_seg = False
    if in_seg:
        fig.add_vrect(
            x0=start, x1=x[-1],
            fillcolor="red", opacity=0.12,
            line_width=0, row=row, col=1,
        )


def plot_speed_with_brakes(driver_df: pd.DataFrame, lap: int):
    """
    Returns a Plotly figure of the speed trace for one lap
    with red shaded brake zones. Used in Introduction and Results pages.
    """
    lap_df = get_lap(driver_df, lap).copy()
    lap_df["time_in_lap_s"] = lap_df[COL_TIME].dt.total_seconds()
    lap_df = (
        lap_df.dropna(subset=["time_in_lap_s", "Speed"])
        .sort_values("time_in_lap_s")
        .reset_index(drop=True)
    )

    fig = go.Figure()

    # Red brake zones
    x_arr = lap_df["time_in_lap_s"].values
    b_arr = lap_df["Brake_active"].values
    in_seg, start = False, None

    for i in range(len(x_arr)):
        if b_arr[i] == 1 and not in_seg:
            start, in_seg = x_arr[i], True
        elif b_arr[i] == 0 and in_seg:
            fig.add_vrect(
                x0=start, x1=x_arr[i],
                fillcolor="red", opacity=0.12, line_width=0,
            )
            in_seg = False
    if in_seg:
        fig.add_vrect(
            x0=start, x1=x_arr[-1],
            fillcolor="red", opacity=0.12, line_width=0,
        )

    # Speed trace
    fig.add_trace(go.Scatter(
        x=x_arr,
        y=lap_df["Speed"].values,
        mode="lines",
        line=dict(color="cornflowerblue", width=2),
        fill="tozeroy",
        fillcolor="rgba(100,149,237,0.08)",
        name="Speed",
    ))

    fig.update_layout(
        xaxis_title="Time within lap (s)",
        yaxis_title="Speed (km/h)",
        title=f"Lap {lap} - red zones = braking",
        height=320,
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig, lap_df
