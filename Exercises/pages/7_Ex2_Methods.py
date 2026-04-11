import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import Exercises.ex2.helper   as helper
import Exercises.ex2.analysis as analysis

st.set_page_config(page_title="Ex2 - Methods", layout="wide")
st.title("Methods - How the Analysis Was Performed")

@st.cache_data
def load():
    return helper.load_signal(), helper.load_events()

sig_df, events_df = load()
fs    = helper.compute_fs(sig_df)
stats = analysis.sampling_stats(sig_df)

#Dataset 
st.markdown("## Dataset at a glance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sampling rate",  f"{stats['fs_median_hz']} Hz")
col2.metric("Duration",       f"{stats['duration_s']} s")
col3.metric("Total samples",  f"{stats['n_samples']:,}")
col4.metric("Event markers",  f"{len(events_df)}")

with st.expander("Raw signal preview (first 200 rows)", expanded=False):
    st.dataframe(sig_df.head(200), use_container_width=True)
    st.caption(f"Shape: {sig_df.shape}   Columns: {list(sig_df.columns)}")

with st.expander("Events table", expanded=False):
    st.dataframe(events_df, use_container_width=True)

#(task 1) Sampling rate
st.markdown(
    """
    ## Task 1 - Determining the sampling rate

    The sampling rate is computed from the **median inter-sample interval**
    of the `time_s` column:

    $$f_s = \\frac{1}{\\text{median}(\\Delta t)}$$

    Using the median (not the mean) makes the estimate robust to any occasional
    timestamp gap or duplicated row.  The Nyquist frequency follows directly:

    $$f_{\\text{Nyquist}} = f_s / 2$$

    Any signal component at a frequency above $f_{\\text{Nyquist}}$ cannot be
    represented faithfully in this recording - it would alias to a lower bin.
    """
)

dt_series = sig_df["time_s"].diff().dropna()
dt_series = dt_series[dt_series > 0]
fs_series = 1.0 / dt_series

fig_fs = go.Figure(go.Histogram(
    x=fs_series.values,
    marker_color="cornflowerblue",
    nbinsx=60,
))
fig_fs.update_layout(
    xaxis_title="Instantaneous sampling rate (Hz)",
    yaxis_title="Count",
    title="Distribution of inter-sample rates - confirms constant-rate recording",
    height=270,
    margin=dict(l=60, r=20, t=50, b=60),
)
st.plotly_chart(fig_fs, use_container_width=True)
st.caption(
    f"Sharp peak at {stats['fs_median_hz']} Hz (std = {stats['fs_std_hz']} Hz). "
    "A narrow distribution confirms the recording is truly constant-rate."
)

#Nyquist table
nyq_df = pd.DataFrame([{
    "fs (Hz)":                stats["fs_median_hz"],
    "Nyquist (Hz)":           stats["nyquist_hz"],
    "Max representable freq": f"{stats['nyquist_hz']} Hz",
    "Implication":            f"Components above {stats['nyquist_hz']} Hz alias into the spectrum",
}])
st.dataframe(nyq_df, use_container_width=True, hide_index=True)

#(task 2) DFT implementation
st.markdown(
    """
    ## Task 2 - DFT implementation from scratch

    The formula:

    $$X[k] = \\sum_{n=0}^{N-1} x[n]\\, e^{-j\\,2\\pi\\,k\\,n\\,/\\,N}$$

    is evaluated as a single matrix-vector multiplication:

    ```python
    def dft(x):
        N = len(x)
        n = np.arange(N)
        k = n[:, None]                        # column vector
        W = np.exp(-2j * np.pi * k * n / N)  # twiddle matrix (N x N)
        return W @ x                          # X[k] for k = 0 … N-1
    ```

    This is $\\mathcal{O}(N^2)$, which is slower than the FFT's
    $\\mathcal{O}(N \\log N)$, but **mathematically identical and fully
    transparent**.  For recordings longer than ~4096 samples we sub-sample
    before transforming to keep computation interactive, using a helper called
    `dft_windowed()`.

    For recordings longer than ~4096 samples the signal is sub-sampled before
    the DFT to keep O(N²) computation interactive, which is a technical necessity,
    not the downsampling analysis of Task 4.
    """
)

#correctness check on a known sinusoid 
st.markdown("### Correctness check on dummy known sinusoid")

fs_v   = 200.0
t_v    = np.linspace(0, 1, int(fs_v), endpoint=False)
f_true = 17.0
x_v    = np.sin(2 * np.pi * f_true * t_v)
X_v    = analysis.dft(x_v)
f_v    = analysis.frequency_bins(len(X_v), fs_v)
mag_v  = analysis.magnitude_spectrum(X_v)

fig_v = make_subplots(
    rows=1, cols=2,
    subplot_titles=(f"Input: {f_true} Hz sine (fs = {fs_v} Hz)",
                    "DFT magnitude - peak must appear at 17 Hz"),
)
fig_v.add_trace(go.Scatter(
    x=t_v[:40], y=x_v[:40], mode="lines",
    line=dict(color="cornflowerblue", width=1.5), name="Signal",
), row=1, col=1)
fig_v.add_trace(go.Scatter(
    x=f_v, y=mag_v, mode="lines",
    line=dict(color="coral", width=1.5),
    fill="tozeroy", fillcolor="rgba(255,127,80,0.10)", name="Magnitude",
), row=1, col=2)
fig_v.add_vline(
    x=f_true, line=dict(color="black", dash="dash", width=1.5),
    annotation_text=f"{f_true} Hz ✓",
    row=1, col=2,
)
fig_v.update_layout(height=300, showlegend=False, margin=dict(l=50, r=20, t=60, b=50))
fig_v.update_xaxes(title_text="Time (s)", row=1, col=1)
fig_v.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
fig_v.update_yaxes(title_text="Amplitude", row=1, col=1)
fig_v.update_yaxes(title_text="|X[k]| / N", row=1, col=2)
st.plotly_chart(fig_v, use_container_width=True)
st.caption(
    f"Our DFT correctly places the single peak at {f_true} Hz "
    "with magnitude ≈ 1.0 (the amplitude of the input sine). ✓"
)

st.markdown(
    """
    ### Noise vs signal classification

    After computing the spectrum, each prominent peak is classified by frequency
    using a simple heuristic: DC, mains interference (50/60 Hz and harmonics),
    or candidate signal. The full classification and reasoning are in the
    Discussion page.
    """
)


st.divider()

# -----------------------------
# Task 3 - Event marker analysis
# -----------------------------

def analyze_event_plateaus(
    sig_df,
    events_df,
    baseline_before_s=1.0,
    search_after_s=5.0,
    smooth_window=21,
    plateau_window_s=0.4,
    plateau_tolerance_fraction=0.15,
):
    df = sig_df.copy().reset_index(drop=True)

    # slight smoothing
    df["amp_smooth"] = (
        df["amplitude"]
        .rolling(window=smooth_window, center=True, min_periods=1)
        .median()
    )

    dt = float(np.median(np.diff(df["time_s"])))
    plateau_n = max(3, int(round(plateau_window_s / dt)))

    results = []

    for i, ev in events_df.iterrows():
        ev_time = float(ev["onset_s"])

        # 1 s before event = baseline window
        base_mask = (
            (df["time_s"] >= ev_time - baseline_before_s) &
            (df["time_s"] < ev_time)
        )

        # 5 s after event = post-event window
        post_mask = (
            (df["time_s"] >= ev_time) &
            (df["time_s"] <= ev_time + search_after_s)
        )

        base_seg = df.loc[base_mask]
        post_seg = df.loc[post_mask].copy()

        if len(base_seg) < 5 or len(post_seg) < plateau_n + 5:
            results.append({
                "event_idx": i,
                "event_time_s": ev_time,
                "baseline": np.nan,
                "low_level": np.nan,
                "drop_depth": np.nan,
                "drop_percent": np.nan,
                "time_to_low_plateau_s": np.nan,
                "low_plateau_duration_s": np.nan,
            })
            continue

        # baseline = median before event
        baseline = float(base_seg["amp_smooth"].median())

        # low level = median of lowest 20% after event
        sorted_vals = np.sort(post_seg["amp_smooth"].values)
        n_low = max(5, int(0.2 * len(sorted_vals)))
        low_level = float(np.median(sorted_vals[:n_low]))

        # drop measures
        drop_depth = baseline - low_level
        drop_percent = 100.0 * drop_depth / baseline if baseline != 0 else np.nan

        # allowed variation around low level
        tol = plateau_tolerance_fraction * drop_depth

        post_vals = post_seg["amp_smooth"].values
        post_times = post_seg["time_s"].values

        plateau_start_idx = None
        plateau_end_idx = None

        # first interval that stays near the low level
        for start in range(0, len(post_seg) - plateau_n + 1):
            window = post_vals[start:start + plateau_n]
            if np.all(np.abs(window - low_level) <= tol):
                plateau_start_idx = start
                break

        if plateau_start_idx is None:
            time_to_low_plateau = np.nan
            low_plateau_duration = np.nan
        else:
            time_to_low_plateau = float(post_times[plateau_start_idx] - ev_time)

            plateau_end_idx = plateau_start_idx + plateau_n - 1
            for j in range(plateau_end_idx + 1, len(post_seg)):
                if np.abs(post_vals[j] - low_level) <= tol:
                    plateau_end_idx = j
                else:
                    break

            low_plateau_duration = float(
                post_times[plateau_end_idx] - post_times[plateau_start_idx]
            )

        results.append({
            "event_idx": i,
            "event_time_s": ev_time,
            "baseline": baseline,
            "low_level": low_level,
            "drop_depth": drop_depth,
            "drop_percent": drop_percent,
            "time_to_low_plateau_s": time_to_low_plateau,
            "low_plateau_duration_s": low_plateau_duration,
        })

    return pd.DataFrame(results)


def plot_annotated_event(
    sig_df,
    events_df,
    event_stats,
    event_idx=4,
    baseline_before_s=1.0,
    search_after_s=5.0,
    smooth_window=21,
):
    ev_time = float(events_df.iloc[event_idx]["onset_s"])
    stats_row = event_stats.iloc[event_idx]

    baseline = stats_row["baseline"]
    low_level = stats_row["low_level"]
    time_to_low_plateau = stats_row["time_to_low_plateau_s"]
    low_plateau_duration = stats_row["low_plateau_duration_s"]

    if pd.isna(baseline) or pd.isna(low_level) or pd.isna(time_to_low_plateau) or pd.isna(low_plateau_duration):
        return None

    plateau_start = ev_time + float(time_to_low_plateau)
    plateau_end = plateau_start + float(low_plateau_duration)

    t_min = ev_time - baseline_before_s - 0.5
    t_max = ev_time + search_after_s + 0.5

    seg = sig_df[(sig_df["time_s"] >= t_min) & (sig_df["time_s"] <= t_max)].copy()

    seg["amp_smooth"] = (
        seg["amplitude"]
        .rolling(window=smooth_window, center=True, min_periods=1)
        .median()
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=seg["time_s"],
        y=seg["amplitude"],
        mode="lines",
        line=dict(color="rgba(100,149,237,0.45)", width=1),
        name="Raw signal",
    ))

    fig.add_trace(go.Scatter(
        x=seg["time_s"],
        y=seg["amp_smooth"],
        mode="lines",
        line=dict(color="cornflowerblue", width=2),
        name="Smoothed signal",
    ))

    fig.add_vrect(
        x0=ev_time - baseline_before_s,
        x1=ev_time,
        fillcolor="lightgreen",
        opacity=0.18,
        line_width=0,
        annotation_text="Baseline window",
        annotation_position="top left",
    )

    fig.add_vrect(
        x0=ev_time,
        x1=ev_time + search_after_s,
        fillcolor="orange",
        opacity=0.10,
        line_width=0,
        annotation_text="Post-event window",
        annotation_position="top right",
    )

    fig.add_vline(
        x=ev_time,
        line=dict(color="tomato", width=2, dash="dash"),
        annotation_text="Event",
        annotation_position="top",
    )

    fig.add_vline(
        x=plateau_start,
        line=dict(color="gold", width=2, dash="dot"),
        annotation_text="Plateau start",
        annotation_position="bottom left",
    )

    fig.add_vline(
        x=plateau_end,
        line=dict(color="gold", width=2, dash="dot"),
        annotation_text="Plateau end",
        annotation_position="bottom right",
    )

    fig.add_hline(
        y=float(baseline),
        line=dict(color="lightgreen", width=2, dash="dash"),
        annotation_text=f"Baseline = {baseline:.3f}",
        annotation_position="right",
    )

    fig.add_hline(
        y=float(low_level),
        line=dict(color="violet", width=2, dash="dash"),
        annotation_text=f"Low level = {low_level:.3f}",
        annotation_position="right",
    )

    fig.add_vrect(
        x0=plateau_start,
        x1=plateau_end,
        fillcolor="violet",
        opacity=0.14,
        line_width=0,
        annotation_text="Low plateau",
        annotation_position="bottom left",
    )

    fig.update_layout(
        title=f"Annotated event analysis at t = {ev_time:.2f} s",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=430,
        margin=dict(l=60, r=20, t=60, b=60),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
    )

    return fig


# compute event stats once
event_stats = analyze_event_plateaus(
    sig_df,
    events_df,
    baseline_before_s=1.0,
    search_after_s=5.0,
    smooth_window=21,
    plateau_window_s=0.4,
    plateau_tolerance_fraction=0.15,
)

st.markdown("## Methods - Example event")

event_idx_plot = st.slider(
    "Select event for annotated example",
    min_value=0,
    max_value=len(events_df) - 1,
    value=min(4, len(events_df) - 1),
    step=1,
)

fig_annot = plot_annotated_event(
    sig_df=sig_df,
    events_df=events_df,
    event_stats=event_stats,
    event_idx=event_idx_plot,
    baseline_before_s=1.0,
    search_after_s=5.0,
    smooth_window=21,
)

if fig_annot is not None:
    st.plotly_chart(fig_annot, use_container_width=True)
    st.caption(
        "Example of the event-based analysis for one selected event. "
        "The plot shows the 1 s baseline window before the marker, the 5 s post-event "
        "analysis window, the estimated baseline, the estimated low level, the time "
        "to low plateau, and the plateau duration."
    )
else:
    st.warning("No valid plateau could be detected for this event with the current settings.")