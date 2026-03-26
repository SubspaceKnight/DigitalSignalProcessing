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

st.set_page_config(page_title="Ex2 - Results", layout="wide")
st.title("Results - Sampling Rate & Frequency Domain Analysis")

@st.cache_data
def load():
    return helper.load_signal(), helper.load_events()

sig_df, events_df = load()
fs    = helper.compute_fs(sig_df)
stats = analysis.sampling_stats(sig_df)

#sidebar
st.sidebar.header("DFT settings")
max_n_dft = st.sidebar.select_slider(
    "Max samples sent to DFT",
    options=[512, 1024, 2048, 4096],
    value=2048,
    help="Higher = more frequency resolution, slower computation (O(N²)).",
)
n_peaks       = st.sidebar.slider("Top peaks to highlight", 3, 20, 10)
freq_limit    = st.sidebar.number_input(
    "Upper frequency limit for plots (Hz)",
    min_value=1.0,
    max_value=float(stats["nyquist_hz"]),
    value=float(stats["nyquist_hz"]),
    step=1.0,
)

#tabs
tab_time, tab_spectrum, tab_events, tab_ds = st.tabs([
    "Time domain",
    "Spectrum (DFT)",
    "Event analysis",
    "Downsampling",
])


#TAB 1 Time domain
with tab_time:
    st.subheader("Task 1 - Raw signal & sampling rate")

    #Sampling rate metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampling rate (fs)",  f"{stats['fs_median_hz']} Hz")
    c2.metric("Sample interval",     f"{stats['dt_mean_ms']} ms")
    c3.metric("Nyquist frequency",   f"{stats['nyquist_hz']} Hz")
    c4.metric("Total samples",       f"{stats['n_samples']:,}")

    #Full signal into every 10th sample 
    sig_thin = sig_df.iloc[::10]
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=sig_thin["time_s"],
        y=sig_thin["amplitude"],
        mode="lines",
        line=dict(color="cornflowerblue", width=0.7),
        name="Signal",
    ))
    #Event onset markers from events_df
    for _, row in events_df.iterrows():
        fig_t.add_vline(
            x=row["onset_s"],
            line=dict(color="tomato", width=1, dash="dot"),
        )
    if not events_df.empty:
        fig_t.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="tomato", dash="dot"),
            name="Event onset",
        ))
    fig_t.update_layout(
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        title=f"Full recording - {stats['n_samples']:,} samples, plotting every 10th ({len(sig_thin):,} pts)",
        height=320, margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig_t, use_container_width=True)
    st.caption(
        "Red dotted lines = event onset markers from events.csv. "
        "Note the baseline drift and amplitude fluctuations - both will appear in the spectrum."
    )

    #Zoom slider
    st.markdown("**Zoom into any window**")
    win_s   = st.slider("Window width (s)", 0.05, min(5.0, float(sig_df["time_s"].max())), 0.5, 0.05)
    t_start = st.slider(
        "Start time (s)", 0.0,
        max(0.0, float(sig_df["time_s"].max()) - win_s),
        0.0, 0.01,
    )
    zoom = sig_df[(sig_df["time_s"] >= t_start) & (sig_df["time_s"] < t_start + win_s)]

    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=zoom["time_s"], y=zoom["amplitude"],
        mode="lines+markers",
        line=dict(color="cornflowerblue", width=1.5),
        marker=dict(size=4),
        name="Signal",
    ))
    for _, row in events_df.iterrows():
        if t_start <= row["onset_s"] <= t_start + win_s:
            fig_z.add_vline(
                x=row["onset_s"],
                line=dict(color="tomato", width=2, dash="dash"),
                annotation_text="Event", annotation_font_color="tomato",
            )
    fig_z.update_layout(
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        height=260, margin=dict(l=60, r=20, t=40, b=60),
        hovermode="x unified",
    )
    st.plotly_chart(fig_z, use_container_width=True)
    st.caption(
        f"Individual samples visible at this zoom level. "
        f"At {stats['fs_median_hz']} Hz, {win_s:.2f} s = ~{int(win_s * stats['fs_median_hz'])} samples."
    )


#TAB 2 Spectrum
with tab_spectrum:
    st.subheader("Task 2 - DFT: magnitude & power spectra")

    @st.cache_data
    def get_spectrum(max_n):
        return analysis.run_full_spectrum(sig_df, fs, max_n=max_n)

    spec  = get_spectrum(max_n_dft)
    freqs = spec["freqs"]
    mag   = spec["magnitude"]

    fmask  = freqs <= freq_limit
    f_plot = freqs[fmask]
    m_plot = mag[fmask]

    if spec["N_raw"] > spec["N_used"]:
        st.info(
            f"Signal has {spec['N_raw']:,} samples. "
            f"DFT computed on {spec['N_used']:,} sub-sampled points "
            f"(effective fs = {spec['fs_used']} Hz) to keep O(N²) tractable. "
            "Increase 'Max samples' in the sidebar for finer resolution."
        )

    #magnitude spectrum is enough for this analysis, but power spectrum is also available in spec["power"] if needed for Task 3 or 4; for later - analysis.run_full_spectrum() for the full DFT pipeline implementation
    st.markdown("### Magnitude spectrum")
    fig_mag = go.Figure()
    fig_mag.add_trace(go.Scatter(
        x=f_plot, y=m_plot, mode="lines",
        line=dict(color="cornflowerblue", width=1.2),
        fill="tozeroy", fillcolor="rgba(100,149,237,0.10)",
        name="|X[k]| / N",
    ))
    fig_mag.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (normalised)",
        title=f"DFT Magnitude Spectrum  (N = {spec['N_used']:,})",
        height=340, margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
    )
    st.plotly_chart(fig_mag, use_container_width=True)


    #peak table 
    st.markdown("### Top spectral peaks")
    peaks_df = analysis.find_spectral_peaks(freqs, mag, n_peaks=n_peaks, min_freq_hz=0.5)
    clf_df   = analysis.classify_peaks(peaks_df)

    #Colour-coded display
    def row_style(r):
        if "Noise" in str(r["Classification"]):
            return ["background-color: rgba(255,80,80,0.12)"] * len(r)
        elif "DC" in str(r["Classification"]):
            return ["background-color: rgba(255,200,80,0.15)"] * len(r)
        else:
            return ["background-color: rgba(60,179,113,0.10)"] * len(r)

    st.dataframe(
        clf_df.style.apply(row_style, axis=1),
        use_container_width=True, hide_index=True,
    )
    st.caption("GREEN Candidate signal  *  RED Noise (mains/harmonic)  *  YELLOW DC / drift") #at the moment only signal




#TODO  TAB 3 - Event analysis
with tab_events:
    def analyze_event_plateaus(
        sig_df,
        events_df,
        baseline_before_s=1.0,
        search_after_s=5.0,
        smooth_window=21,
        plateau_window_s=0.4,
        plateau_tolerance_fraction=0.15,
    ):
        """
        Analyze event-related amplitude drops and low plateaus.

        Returns one row per event with:
        - baseline
        - low_level
        - drop_depth
        - drop_percent
        - time_to_low_plateau_s
        - low_plateau_duration_s
        """

        df = sig_df.copy().reset_index(drop=True)
        df["amp_smooth"] = (
            df["amplitude"]
            .rolling(window=smooth_window, center=True, min_periods=1)
            .median()
        )

        dt = float(np.median(np.diff(df["time_s"])))
        plateau_n = max(3, int(round(plateau_window_s / dt)))

        rows = []

        for i, ev in events_df.iterrows():
            ev_time = float(ev["onset_s"])

            base_mask = (
                (df["time_s"] >= ev_time - baseline_before_s) &
                (df["time_s"] < ev_time)
            )
            post_mask = (
                (df["time_s"] >= ev_time) &
                (df["time_s"] <= ev_time + search_after_s)
            )

            base_seg = df.loc[base_mask]
            post_seg = df.loc[post_mask].copy()

            if len(base_seg) < 5 or len(post_seg) < plateau_n + 5:
                rows.append({
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

            baseline = float(base_seg["amp_smooth"].median())

            # Estimate low level from the lowest 20% of post-event values
            sorted_vals = np.sort(post_seg["amp_smooth"].values)
            n_low = max(5, int(0.2 * len(sorted_vals)))
            low_level = float(np.median(sorted_vals[:n_low]))

            drop_depth = baseline - low_level
            drop_percent = 100.0 * drop_depth / baseline if baseline != 0 else np.nan

            tol = plateau_tolerance_fraction * drop_depth

            post_vals = post_seg["amp_smooth"].values
            post_times = post_seg["time_s"].values

            plateau_start_idx = None
            plateau_end_idx = None

            # Find first stable low plateau
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

                # Continue from plateau start until signal clearly leaves plateau
                plateau_end_idx = plateau_start_idx + plateau_n - 1

                for j in range(plateau_end_idx + 1, len(post_seg)):
                    if np.abs(post_vals[j] - low_level) <= tol:
                        plateau_end_idx = j
                    else:
                        break

                low_plateau_duration = float(
                    post_times[plateau_end_idx] - post_times[plateau_start_idx]
                )

            rows.append({
                "event_idx": i,
                "event_time_s": ev_time,
                "baseline": baseline,
                "low_level": low_level,
                "drop_depth": drop_depth,
                "drop_percent": drop_percent,
                "time_to_low_plateau_s": time_to_low_plateau,
                "low_plateau_duration_s": low_plateau_duration,
            })

        return pd.DataFrame(rows)

    event_stats = analyze_event_plateaus(
        sig_df,
        events_df,
        baseline_before_s=1.0,
        search_after_s=5.0,
        smooth_window=21,
        plateau_window_s=0.4,
        plateau_tolerance_fraction=0.15,
    )

    st.dataframe(event_stats, use_container_width=True)

    valid = event_stats.dropna()

    st.markdown("## Summary")

    summary_df = pd.DataFrame({
        "Metric": [
            "Drop depth",
            "Drop percent",
            "Time to low plateau (s)",
            "Low plateau duration (s)",
        ],
        "Mean": [
            valid["drop_depth"].mean(),
            valid["drop_percent"].mean(),
            valid["time_to_low_plateau_s"].mean(),
            valid["low_plateau_duration_s"].mean(),
        ],
        "SD": [
            valid["drop_depth"].std(),
            valid["drop_percent"].std(),
            valid["time_to_low_plateau_s"].std(),
            valid["low_plateau_duration_s"].std(),
        ]
    })

    st.dataframe(summary_df, use_container_width=True)

#TODO  TAB 4 Downsampling  (Task 4 - not yet implemented)
with tab_ds:
    st.info(
        "**Task 4 - Downsampling** has not been implemented yet.\n\n"
        "Planned: downsample the signal at multiple factors (without filtering), "
        "show the effect on the time-domain waveform and the frequency spectrum, "
        "and explain when and where aliasing arises."
    )
st.divider()
st.caption("DSP Exercise 2 * FH Joanneum * 2026")