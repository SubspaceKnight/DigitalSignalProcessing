import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from Exercises.utils.shared import compute_fs_stats 

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR.parent / "data"
SIGNAL_PATH = DATA_DIR / "signal.csv"
EVENTS_PATH = DATA_DIR / "events.csv"

@st.cache_data
def load_signal(path: Path = SIGNAL_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
    df = df.rename(columns={"signal": "amplitude"})

    if "event_marker" in df.columns:
        df["event_marker"] = (
            pd.to_numeric(df["event_marker"], errors="coerce")
            .fillna(0).astype(int)
        )

    return (
        df.dropna(subset=["time_s", "amplitude"])
        .sort_values("time_s")
        .reset_index(drop=True)
    )


@st.cache_data
def load_events(path: Path = EVENTS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={"marker_time_s": "onset_s"})
    df["onset_s"]  = pd.to_numeric(df["onset_s"],  errors="coerce")
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")

    return (
        df.dropna(subset=["onset_s"])
        .sort_values("onset_s")
        .reset_index(drop=True)
    )


def compute_fs(sig_df: pd.DataFrame) -> float:
    return compute_fs_stats(sig_df["time_s"])["median_hz"]


def downsample(sig_df: pd.DataFrame, factor: int) -> pd.DataFrame:
    if factor < 1:
        raise ValueError("Downsample factor must be >= 1.")
    return sig_df.iloc[::factor].reset_index(drop=True).copy()