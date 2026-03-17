import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

DATA_PATH = Path("Bahrain_time_series.csv")

COL_LAP     = "LapNumber"
COL_TIME    = "Time"
COL_SESSION = "SessionTime"
COL_DRIVER  = "Driver"          
COL_ID      = "ID"            

DRIVER_CODE = "VER"
DRIVER_ID   = 0

#all numeric telemetry channels
TELEMETRY_COLS = ["Speed", "RPM", "Throttle", "Brake", "nGear", "DRS", "DistanceToDriverAhead", "X", "Y"]

#Loaders
@st.cache_data
def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    for col in ["Time", "SessionTime"]:
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col], errors="coerce")

    if "Brake" in df.columns:
        df["Brake_active"] = (
            df["Brake"].astype(str).str.strip().str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0).astype(int)
        )
    else:
        df["Brake_active"] = 0

    for col in ["LapNumber", "Speed", "RPM", "Throttle", "nGear"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_driver(df: pd.DataFrame, driver_code: str = DRIVER_CODE) -> pd.DataFrame:
    #print(f"Available drivers: {df[COL_DRIVER].unique()}")
    #print(f"Driver {driver_code} has {len(df[df[COL_DRIVER] == driver_code])} rows.")
    return df[df[COL_DRIVER] == driver_code].copy()


#Lap-level helpers
def list_laps(driver_df: pd.DataFrame) -> list[int]:
    return sorted(driver_df[COL_LAP].unique().tolist())


def get_lap(driver_df: pd.DataFrame, lap: int) -> pd.DataFrame:
    return driver_df[driver_df[COL_LAP] == lap].copy()


#Normalization 
def normalize_lap_to_percent(lap_df: pd.DataFrame, signal: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a single lap's signal onto 0-100% lap completion. This solves the 'laps have different lengths' problem: we interpolate every lap to the same 500-point grid so they can be directly overlaid and averaged.
    """
    n_points = 500
    x_orig   = np.linspace(0, 100, len(lap_df))
    x_new    = np.linspace(0, 100, n_points)
    y_new    = np.interp(x_new, x_orig, lap_df[signal].values)
    return x_new, y_new


def build_lap_matrix(driver_df: pd.DataFrame, signal: str) -> np.ndarray:
    """
    2-D array of shape (n_laps, 500). Each row is one lap normalized to 0-100%. Columns = lap completion percentage.
    """
    laps = list_laps(driver_df)
    rows = []
    for lap in laps:
        lap_df = get_lap(driver_df, lap)
        if len(lap_df) < 10:          # skip suspiciously short laps if any
            continue
        _, y = normalize_lap_to_percent(lap_df, signal)
        rows.append(y)
    return np.array(rows), [l for l in laps if len(get_lap(driver_df, l)) >= 10]


#Statistics 
def lap_summary(driver_df: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Per-lap descriptive stats for one signal."""
    return ( driver_df.groupby(COL_LAP)[signal].agg(mean="mean", std="std", min="min", max="max", median="median").reset_index().rename(columns={COL_LAP: "Lap"}) )


def flag_outlier_laps(driver_df: pd.DataFrame, signal="Speed", z_threshold=2) -> list[int]:
    """
    Flag laps that contain any track status other than:
      '1' = track clear
      '2' = yellow flag
    Everything else (e.g. safety car, red flag) is treated as an outlier.
    """
    #print("Unique TrackStatus values:", sorted(driver_df["TrackStatus"].unique().tolist()))
    track_status = pd.to_numeric(driver_df["TrackStatus"], errors="coerce")
    bad_laps = (
        driver_df.loc[~track_status.isin([1, 2]), COL_LAP]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return sorted(bad_laps)

if __name__ == "__main__":
    df = load_raw()
    driver_df = get_driver(df)
    print(driver_df.head())
    print(lap_summary(driver_df, "Speed"))
    print("Outlier laps:", flag_outlier_laps(driver_df, "Speed"))
