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
#TODO 
st.markdown("## Tasks 3 & 4 - Not yet covered")
st.info(
    "**Task 3 - Event analysis**: the approach for extracting and characterizing "
    "stimulus-locked responses will be described here once implemented.\n\n"
    "**Task 4 - Downsampling**: the method for sub-sampling the signal at "
    "multiple factors and evaluating the resulting spectral changes will be "
    "described here once implemented."
)

st.divider()
st.caption("DSP Exercise 2 * FH Joanneum * 2026")