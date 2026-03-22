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
    "Event analysis \u2014 not yet solved",
    "Downsampling \u2014 not yet solved",
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





#TODO  TAB 3 - Event analysis  (Task 3 - not yet implemented)
with tab_events:
    st.info(
        "**Task 3 - Event analysis** has not been implemented yet.\n\n"
        "Planned: hypothesize about event stimuli, measure event duration, "
        "characterize the signal response locked to each event marker."
    )

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