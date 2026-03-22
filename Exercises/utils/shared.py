import numpy as np
import pandas as pd
import plotly.graph_objects as go

COLORS = {
    "blue":       "cornflowerblue",
    "red":        "tomato",
    "green":      "mediumseagreen",
    "orange":     "coral",
    "purple":     "mediumpurple",
    "gold":       "goldenrod",
    "grey":       "rgba(180,180,180,0.6)",
    "fill_blue":  "rgba(100,149,237,0.10)",
    "fill_red":   "rgba(255,127,80,0.10)",
    "fill_green": "rgba(60,179,113,0.10)",
}


#Resampling
def resample_to_grid(y: np.ndarray, n_points: int = 500) -> np.ndarray:
    y      = np.asarray(y, dtype=float)
    x_orig = np.linspace(0.0, 1.0, len(y))
    x_new  = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_orig, y)


#Sampling-rate statistics
def compute_fs_stats(time_seconds: pd.Series) -> dict:
    dt = time_seconds.diff().dropna()
    dt = dt[dt > 0]                       #drop resets / identical timestamps
    fs = 1.0 / dt
    return {
        "mean_hz":   round(float(fs.mean()),   4),
        "median_hz": round(float(fs.median()), 4),
        "std_hz":    round(float(fs.std()),    4),
        "dt_mean_ms":round(float(dt.mean()) * 1000, 4),
        "nyquist_hz":round(float(fs.median()) / 2, 4),
        "n_samples": len(time_seconds),
        "duration_s":round(float(time_seconds.max() - time_seconds.min()), 4),
    }


#Boolean-segment detection
def find_boolean_segments(arr: np.ndarray) -> list[tuple[int, int]]:
    arr  = np.asarray(arr, dtype=int)
    diff = np.diff(arr, prepend=0, append=0)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0] - 1   #inclusive
    return list(zip(starts.tolist(), ends.tolist()))


#Plotly vrect helper
def add_vrect_segments(fig: go.Figure, x_arr: np.ndarray, binary_arr: np.ndarray, fillcolor: str  = "red", opacity: float  = 0.12, row: int | None = None, col: int | None = None,) -> None:
    x_arr      = np.asarray(x_arr,      dtype=float)
    binary_arr = np.asarray(binary_arr, dtype=int)

    kwargs: dict = dict(fillcolor=fillcolor, opacity=opacity, line_width=0)
    if row is not None:
        kwargs["row"] = row
    if col is not None:
        kwargs["col"] = col

    for start_idx, end_idx in find_boolean_segments(binary_arr):
        x0 = float(x_arr[start_idx])
        #the sample *after* the segment end as the right edge if available
        x1 = float(x_arr[end_idx + 1]) if end_idx + 1 < len(x_arr) else float(x_arr[end_idx])
        fig.add_vrect(x0=x0, x1=x1, **kwargs)