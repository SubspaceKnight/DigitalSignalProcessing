import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import Exercises.ex2.helper   as helper
import Exercises.ex2.analysis as analysis

st.set_page_config(page_title="Ex2 - Discussion", layout="wide")
st.title("Discussion - Sampling Rate, Nyquist, and Spectral Findings")

@st.cache_data
def load():
    return helper.load_signal(), helper.load_events()

sig_df, events_df = load()
fs    = helper.compute_fs(sig_df)
stats = analysis.sampling_stats(sig_df)

@st.cache_data
def get_spec(max_n=2048):
    return analysis.run_full_spectrum(sig_df, fs, max_n=max_n)

spec     = get_spec()
peaks_df = analysis.find_spectral_peaks(spec["freqs"], spec["magnitude"], n_peaks=15, min_freq_hz=0.1)
clf_df   = analysis.classify_peaks(peaks_df)

#(task 1) Sampling rate & Nyquist discussion
st.markdown("## Task 1 – Sampling rate & the Nyquist criterion")
 
#Pre-compute values to avoid f-string / LaTeX brace conflicts
_dt     = stats['dt_mean_ms']
_fs     = stats['fs_median_hz']
_fs_std = stats['fs_std_hz']
_n      = f"{stats['n_samples']:,}"
_nyq    = stats['nyquist_hz']
_df     = round(stats['fs_median_hz'] / stats['n_samples'], 4)
_dur    = stats['duration_s']
 
st.markdown("### What the sampling rate is")
st.markdown(
    f"The inter-sample interval is computed from the `time_s` column "
    f"using the median (robust to any timestamp glitches):"
)
st.latex(
    r"\Delta t = " + str(_dt) + r"\,\text{ms}"
    r"\quad \Longrightarrow \quad "
    r"f_s = \frac{1}{\Delta t} = " + str(_fs) + r"\,\text{Hz}"
)
st.markdown(
    f"The sampling rate is **very stable** (std = {_fs_std} Hz across {_n} samples), "
    f"confirming a constant-rate acquisition system."
)
 
st.markdown("### What this implies for the Nyquist criterion")
st.markdown(
    "The **Nyquist-Shannon theorem** states that a bandlimited signal can be "
    "perfectly reconstructed if sampled at $f_s \\geq 2 f_{\\max}$. "
    "The maximum frequency we can faithfully represent is therefore:"
)
st.latex(
    r"f_{\text{Nyquist}} = \frac{f_s}{2} = \frac{1}{2\,\Delta t} = \frac{1}{2 \times "
    + str(_dt) + r"\,\text{ms}} = " + str(_nyq) + r"\,\text{Hz}"
)
st.markdown(
    f"**Implications for this recording:**\n\n"
    f"- Any genuine signal component must have $f < {_nyq}$ Hz.\n"
    f"- Frequency resolution of the DFT: $\\Delta f = f_s / N = {_df}$ Hz "
    f"(very fine — recording is {_dur} s long).\n"
    f"- Energy above {_nyq} Hz cannot be represented and would alias into the "
    f"spectrum. Since this is a controlled recording, the signal is assumed "
    f"bandwidth-limited at the sensor level."
)
 
c1, c2, c3, c4 = st.columns(4)
c1.metric("fs",                f"{stats['fs_median_hz']} Hz")
c2.metric("Nyquist",           f"{stats['nyquist_hz']} Hz")
c3.metric("Duration",          f"{stats['duration_s']} s")
c4.metric("Frequency resolution Δf", f"{round(stats['fs_median_hz'] / stats['n_samples'], 4)} Hz")


#(task 2) Signal vs noise discussion
st.markdown("## Task 2 - Signal vs noise in the spectrum")

st.markdown("### Classified spectral peaks")

#Full annotated table
st.dataframe(clf_df, use_container_width=True, hide_index=True)

#Summary counts
n_noise  = clf_df["Classification"].str.contains("Noise").sum()
n_dc     = clf_df["Classification"].str.contains("DC").sum()
n_signal = clf_df["Classification"].str.contains("signal").sum()

c1, c2, c3 = st.columns(3)
c1.metric("Noise peaks found",         n_noise)
c2.metric("DC / drift components",     n_dc)
c3.metric("Candidate signal peaks",    n_signal)

#annotated plot 
freqs  = spec["freqs"]
mag    = spec["magnitude"]
fmask  = freqs <= float(stats["nyquist_hz"])
f_plot = freqs[fmask]
m_plot = mag[fmask]

fig_ann = go.Figure()
fig_ann.add_trace(go.Scatter(
    x=f_plot, y=m_plot, mode="lines",
    line=dict(color="cornflowerblue", width=1.2),
    fill="tozeroy", fillcolor="rgba(100,149,237,0.08)",
    name="Magnitude",
))
for _, r in clf_df.iterrows():
    color = (
        "tomato"        if "Noise"  in str(r["Classification"]) else
        "goldenrod"     if "DC"     in str(r["Classification"]) else
        "mediumseagreen"
    )
    fig_ann.add_vline(
        x=r["Frequency (Hz)"],
        line=dict(color=color, dash="dot", width=1.2),
        annotation_text=f"{r['Frequency (Hz)']} Hz",
        annotation_font=dict(color=color, size=9),
    )
fig_ann.update_layout(
    xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
    title="Annotated spectrum — green = signal · red = noise · gold = DC",
    height=360, margin=dict(l=60, r=20, t=50, b=60),
    hovermode="x unified",
)
st.plotly_chart(fig_ann, use_container_width=True)

#noise source discussion 
st.markdown(
    """
    ### Noise sources and their frequencies

    #### Power-line interference (50 / 60 Hz and harmonics)
    Mains power at 50 Hz (Europe) or 60 Hz (North America) couples into any
    recording system through capacitive or inductive pickup on cables, ground
    loops, or insufficiently shielded amplifiers.  The waveform on the mains
    is not a perfect sinusoid - it contains harmonic distortion - so the noise
    appears not only at the fundamental but also at 100/150/200 Hz (for 50 Hz)
    or 120/180/240 Hz (for 60 Hz).  These peaks are **narrow and stable**,
    making them the easiest noise to identify.

    #### DC component (0 Hz)
    A non-zero mean in the signal appears as a spike at 0 Hz.  In electrode-based
    recordings this typically reflects a half-cell potential offset at the
    electrode-tissue interface, or a DC bias in the amplifier chain.  It carries
    no meaningful signal information and is usually removed by high-pass filtering
    before analysis.

    #### Broadband noise floor
    The roughly flat magnitude level across the entire spectrum (visible between
    peaks) is **broadband noise** - thermal (Johnson-Nyquist) noise from
    resistances in the circuit, quantisation noise from the ADC, and
    electromagnetic interference that does not concentrate at a single frequency.
    This floor sets the minimum detectable signal amplitude.

    ### What is actual signal?
    Peaks that are **not** explained by mains frequencies, their harmonics, or DC
    are candidate signal components.  In a physiological or neuroscientific
    context these would reflect rhythmic biological processes (e.g. alpha/beta
    brain oscillations, cardiac rhythm, respiratory modulation) or a response
    driven by a presented stimulus.  Without additional domain knowledge the
    assignment can only be *probabilistic* - but the frequency, stability, and
    relationship to the event markers (addressed in Task 3) help disambiguate.

    """
)

#summary 
st.markdown("## Summary")
st.markdown(
    f"""
    | Question | Answer |
    |----------|--------|
    | Sampling rate | **{stats['fs_median_hz']} Hz** |
    | Nyquist frequency | **{stats['nyquist_hz']} Hz** |
    | Duration | {stats['duration_s']} s ({stats['n_samples']:,} samples) |
    | Frequency resolution | {round(stats['fs_median_hz'] / stats['n_samples'], 4)} Hz |
    | Dominant noise | Mains interference ({n_noise} peak(s) classified as noise) |
    | DC component | {"Present" if n_dc > 0 else "Not detected"} |
    | Candidate signal peaks | {n_signal} peak(s) not attributable to known noise |
    | DFT implementation | From-scratch twiddle-matrix multiplication - no library FFT used |
    """
)


st.divider()
#TODO
st.markdown("## Task 3 - Event analysis *(not yet solved)*")
st.info(
    "This section will discuss what is happening during the marked events: "
    "hypothesized stimulus type, event duration, response characteristics, "
    "and inter-event intervals."
)

st.markdown("## Task 4 - Downsampling *(not yet solved)*")
st.info(
    "This section will discuss the effect of downsampling without an "
    "anti-aliasing filter on both the time-domain waveform and the spectrum, "
    "including when aliasing arises and how to predict aliased peak positions."
)

st.divider()
st.caption("DSP Exercise 2 • FH Joanneum • 2026")