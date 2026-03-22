import streamlit as st
import numpy as np
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

st.set_page_config(page_title="Ex2 - Introduction", layout="wide")
st.title("Introduction - Frequency Domain Analysis & the DFT")

@st.cache_data
def load():
    return helper.load_signal(), helper.load_events()

sig_df, events_df = load()
fs    = helper.compute_fs(sig_df)
stats = analysis.sampling_stats(sig_df)


#the data we are working with...
st.markdown("## The data we are working with")
st.markdown(
    f"The recording is **{stats['duration_s']} s** long, sampled at "
    f"**{stats['fs_median_hz']} Hz** ({stats['n_samples']:,} samples total). "
    f"Below: the full signal on the left, one second zoom on the right."
)

#Every 10th sample (~46 k pts)
sig_thin = sig_df.iloc[::10]

fig_full = go.Figure()
fig_full.add_trace(go.Scatter(
    x=sig_thin["time_s"],
    y=sig_thin["amplitude"],
    mode="lines",
    line=dict(color="cornflowerblue", width=0.7),
    name="Signal",
))
#150 event markers
for _, ev in events_df.iterrows():
    fig_full.add_vline(
        x=ev["onset_s"],
        line=dict(color="tomato", width=1, dash="dot"),
    )
fig_full.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    title=(
        f"Full recording  —  {stats['n_samples']:,} samples total, "
        f"plotting every 10th ({len(sig_thin):,} pts)  *  "
        f"{len(events_df)} event markers"
    ),
    height=340,
    margin=dict(l=60, r=20, t=50, b=60),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02),
)
st.plotly_chart(fig_full, use_container_width=True)
st.caption(
    f"460,800 raw samples * every 10th plotted ({len(sig_thin):,} points) * "
    f"red dotted lines = {len(events_df)} event markers. "
    "No information is lost visually: at this scale adjacent samples overlap."
)


st.divider()

st.markdown(
    """
    ## What is the Frequency Domain?

    Every signal recorded in the time domain - a sequence of amplitude values
    at evenly spaced moments - can equivalently be described as a **sum of
    sinusoids**, each with its own frequency, amplitude, and phase.

    The **Discrete Fourier Transform (DFT)** performs this decomposition.
    Given a finite sequence $x[0], x[1], \\ldots, x[N-1]$ it produces:

    $$
    X[k] = \\sum_{n=0}^{N-1} x[n]\\, e^{-j\\, 2\\pi\\, k\\, n\\, /\\, N},
    \\qquad k = 0, 1, \\ldots, N-1
    $$

    - $k$ indexes the **frequency bin**.  Physical frequency: $f_k = k \\cdot f_s / N$.
    - Because $x[n]$ is real, only the first $N/2 + 1$ bins are unique
      (conjugate symmetry) - the rest mirror them.
    """
)

#DFT of the actual signal 
st.markdown("### What the DFT reveals - magnitude spectrum of the actual signal")

spec_intro = analysis.run_full_spectrum(sig_df, fs, max_n=2048)
f_intro    = spec_intro["freqs"]
m_intro    = spec_intro["magnitude"]

threshold = st.slider(
    "Magnitude threshold - highlight peaks above this level",
    min_value=0.001, max_value=0.5,
    value=0.03, step=0.001, format="%.3f",
)

above_mask = m_intro >= threshold
f_above    = f_intro[above_mask]
m_above    = m_intro[above_mask]

fig_spec = go.Figure()

#Full spectrum trace
fig_spec.add_trace(go.Scatter(
    x=f_intro, y=m_intro, mode="lines",
    line=dict(color="cornflowerblue", width=1.2),
    fill="tozeroy", fillcolor="rgba(100,149,237,0.12)",
    name="Magnitude",
))

#Peaks above threshold are highlighted as scatter markers
if len(f_above) > 0:
    fig_spec.add_trace(go.Scatter(
        x=f_above, y=m_above,
        mode="markers",
        marker=dict(color="tomato", size=7, symbol="circle"),
        name=f"Above threshold ({len(f_above)} peaks)",
        hovertemplate="%{x:.4f} Hz  |  mag = %{y:.4f}<extra></extra>",
    ))

#Threshold horizontal line
fig_spec.add_hline(
    y=threshold,
    line=dict(color="tomato", dash="dash", width=1.5),
    annotation_text=f"threshold = {threshold:.3f}",
    annotation_font=dict(color="tomato"),
    annotation_position="top right",
)

fig_spec.update_layout(
    xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (normalised)",
    title=f"DFT magnitude spectrum — {len(f_above)} component(s) above threshold {threshold:.3f}",
    height=340, margin=dict(l=60, r=20, t=50, b=60),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02),
)
st.plotly_chart(fig_spec, use_container_width=True)
st.caption(
    "Drag the slider above to change the threshold. "
    "Red dots = components above the threshold — these are the peaks worth investigating. "
    "The flat region below the threshold is the broadband noise floor."
)

st.divider()

#TODO do we need to cover anything for tasks 3 & 4 here? Or just say "will be added in the next iteration" and leave it at that?
st.markdown("## Tasks 3 & 4 - Not yet covered")
st.info(
    "**Task 3 - Event analysis** and **Task 4 - Downsampling** will be added "
    "in the next iteration.\n\n"
    "Task 3 will introduce epoch analysis and the concept of stimulus-locked "
    "responses (ERP / event-related activity).\n\n"
    "Task 4 will explain downsampling, the resulting change in Nyquist frequency, "
    "and how aliasing arises when no anti-aliasing filter is applied beforehand."
)

st.divider()
st.caption("DSP Exercise 2 * FH Joanneum * 2026")