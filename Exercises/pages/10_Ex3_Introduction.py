import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent  # -> Exercises
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ex3 import helper, analysis

st.set_page_config(page_title="Ex3 - Introduction", layout="wide")
st.title("Introduction - Time-Frequency Analysis & the STFT")

#Motivation
st.markdown(
    """
    ## Why time-frequency analysis?

    Both approaches covered so far have a fundamental limitation:

    - **Time domain** — tells us *when* things happen, but not *what frequencies* are present - they are mixed together.
    - **Frequency domain (DFT)** — reveals the full spectral content, but collapses the entire
      recording into one spectrum. We lose all temporal information: a chord played at the
      beginning looks identical to the same chord played at the end.

    Music, speech, and most real-world signals are **non-stationary** — their frequency content
    changes over time. The Short-Time Fourier Transform (STFT) resolves this by computing a
    local DFT over a sliding window, giving us a two-dimensional representation:
    **frequency & time**.
    """
)

st.divider()

# STFT concept
st.markdown(
    r"""
    ## The Short-Time Fourier Transform

    The STFT divides the signal into short, overlapping frames and applies the DFT to each frame:

    $$
    X[k, m] = \sum_{n=0}^{L-1} x[mH + n] \cdot w[n] \cdot e^{-j\,2\pi\,k\,n\,/\,L}
    $$

    where:
    - $L$ = window length (samples)
    - $H$ = hop length (samples) — controls overlap: overlap = $L - H$
    - $w[n]$ = window function
    - $m$ = frame index
    - $k$ = frequency bin index → physical frequency $f_k = k \cdot f_s / L$

    The result is a **complex matrix** $X[k, m]$ of shape $(L/2 + 1) \times M$,
    where $M$ is the number of frames.
    """
)

st.divider()

#Time-frequency trade-off — interactive demo
st.markdown(
    """
    ## The time-frequency trade-off

    The window length $L$ creates a fundamental trade-off:

    | Long window | Short window |
    |---|---|
    | Fine frequency resolution ($\Delta f = f_s / L$ small) | Coarse frequency resolution |
    | Coarse time resolution ($\Delta t = L / f_s$ large) | Fine time resolution |
    | Good for sustained tones | Good for transients (drums, clicks) |

    This is a consequence of the **Heisenberg-Gabor uncertainty principle** for signals:
    we cannot simultaneously achieve perfect time and frequency resolution.
    """
)

st.markdown("### Interactive demo — effect of window length on a two-tone signal")

st.markdown(
    """
    The signal below consists of two pure sinusoids — **440 Hz** (A4, a musical "A") and
    **880 Hz** (A5, one octave higher, at 70 % amplitude). Because both tones are constant
    throughout the recording, the spectrogram should ideally show two thin, perfectly
    horizontal lines. How well it does that depends entirely on your window settings.

    **Reading the color axis (dB scale)**

    The color encodes *magnitude in decibels*:
    $$\\text{dB}[k,m] = 20\\log_{10}|X[k,m]|$$
    Decibels are logarithmic: every +20 dB corresponds to x10 in amplitude. The scale is
    anchored so that the loudest bin in the spectrogram is **0 dB** (white/yellow), and
    anything 60 dB quieter appears dark blue/black. Bins below -60 dB are treated as silence
    and clipped to the bottom of the range — this keeps noise from washing out the display.

    **What to look for**
    - With a **long window** the two frequency lines appear sharp and narrow → good frequency
      resolution (small Δf), but each column covers a longer stretch of time.
    - With a **short window** the lines blur vertically (large Δf) while each column
      represents a shorter time slice → good time resolution, poor frequency resolution.
    - Switching from *Rectangular* to a smooth window (Hann, Hamming, Blackman) visibly
      suppresses the faint horizontal smear between the two main lines — those are **spectral
      leakage** artefacts caused by the abrupt frame edges of the rectangular window.
    - The faint glow at the **very bottom of the plot (0-100 Hz)** is also a leakage
      artefact, not real signal energy. The k = 0 bin (DC) measures the mean of each
      windowed frame — theoretically zero for a zero-mean sinusoid, but floating-point
      summation leaves a tiny residual that becomes visible on a dB scale. The near-DC bins
      pick up side-lobe energy leaking *down* from 440 Hz: the rectangular window's first
      side-lobe is only -13 dB below the main lobe, so this low-frequency contamination is
      substantial. Switch to Hann or Blackman and it almost completely disappears — a direct
      demonstration of why window choice matters.
    """
)

fs_demo = 2048*3
t_demo = np.linspace(0, 1.0, fs_demo, endpoint=False)
#Two tones: 440 Hz and 880 Hz
x_demo = np.sin(2 * np.pi * 440 * t_demo) + 0.7 * np.sin(2 * np.pi * 880 * t_demo)

col_sl1, col_sl2 = st.columns(2)
with col_sl1:
    win_len_demo = st.select_slider(
        "Window length (samples)",
        options=[64, 128, 256, 512, 1024, 2048],
        value=512,
    )
with col_sl2:
    win_fn_demo = st.selectbox(
        "Window function",
        options=analysis.WINDOW_FUNCTIONS,
        index=0,
    )

hop_demo = win_len_demo // 2
result_demo = analysis.stft(
    x_demo, fs_demo,
    window_name=win_fn_demo,
    win_length=win_len_demo,
    hop_length=hop_demo,
)
mag_db_demo = analysis.to_db(result_demo["magnitude"])

# Widen to 80 dB so rectangular-window leakage (~-13 dB below peak) falls
# into the darker portion of Magma rather than the mid-range red.
DB_RANGE = 80
z_max = mag_db_demo.max()
z_min = z_max - DB_RANGE
mag_db_demo[mag_db_demo < z_min] = z_min

fig_demo = go.Figure(go.Heatmap(
    x=result_demo["times"],
    y=result_demo["freqs"],
    z=mag_db_demo,
    colorscale="Magma",
    zmin=z_min-10, #workaround against the artifacts on the edges of the plot
    zmax=z_max,
    colorbar=dict(
        title="rel. dB",
        tickvals=[z_min, z_min + DB_RANGE * 0.25, z_min + DB_RANGE * 0.5,
                  z_min + DB_RANGE * 0.75, z_max],
        ticktext=[f"-{DB_RANGE} dB", f"-{DB_RANGE // 4 * 3} dB",
                  f"-{DB_RANGE // 2} dB", f"-{DB_RANGE // 4} dB",
                  "peak"], 
    ),
))
fig_demo.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Frequency (Hz)",
    yaxis_range=[0, 2000],
    title=(
        f"STFT spectrogram — window = {win_len_demo} samples "
        f"({win_len_demo / fs_demo * 1000:.1f} ms), "
        f"Δf = {fs_demo / win_len_demo:.1f} Hz, "
        f"window = {win_fn_demo}"
    ),
    height=360,
    margin=dict(l=60, r=20, t=50, b=60),
    plot_bgcolor="#0d0221",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cccccc"),
)
st.plotly_chart(fig_demo, use_container_width=True, theme=None)
st.caption(
    "440 Hz + 880 Hz two-tone signal, 1 s at 8 kHz. "
    "Color = relative magnitude: white/yellow = loudest bin (labeled 'peak'), dark = -80 dB below peak. "
    "The faint glow near 0 Hz is a DC/leakage artefact — try switching to Hann to suppress it. "
    "Try: Rectangular → Hann to see leakage reduction; "
    "64 → 2048 samples to see the frequency-resolution trade-off."
)

st.divider()

#Window functions
st.markdown(
    """
    ## Window functions

    Multiplying each frame by a window function before the DFT reduces **spectral leakage**.
    A rectangular window abruptly cuts the signal at the frame edges, producing strong
    side-lobes in the spectrum (the Gibbs phenomenon). Smooth windows taper to zero at both
    ends and suppress side-lobes at the cost of a slightly wider main lobe.

    | Window | Main lobe width | Side-lobe level | Use case |
    |        |                 |                 |          |
    | Rectangular | Narrowest | -13 dB (high) | Not recommended for audio |
    | Hann | Medium | -31 dB | General-purpose audio analysis |
    | Hamming | Medium | -41 dB | Speech processing |
    | Blackman | Widest | -57 dB (very low) | When side-lobe suppression matters most |
    """
)

#Plot all window shapes side by side
L_win = 256
fig_win = go.Figure()
colors = ["cornflowerblue", "coral", "mediumseagreen", "goldenrod"] 
for name, color in zip(analysis.WINDOW_FUNCTIONS, colors):
    w = analysis.get_window(name, L_win)
    fig_win.add_trace(go.Scatter(
        x=np.arange(L_win), y=w,
        mode="lines", name=name,
        line=dict(color=color, width=1.8),
    ))
fig_win.update_layout(
    xaxis_title="Sample index n",
    yaxis_title="Amplitude",
    title=f"Window shapes (L = {L_win} samples)",
    height=280,
    margin=dict(l=60, r=20, t=50, b=60),
    legend=dict(orientation="h", y=1.02),
)
st.plotly_chart(fig_win, use_container_width=True)

st.divider()

#Y-axis representations
st.markdown(
    """
    ## Y-axis representations of the spectrogram

    The STFT output $X[k, m]$ can be visualised in several ways:

    ### 1. Magnitude / Power spectrum
    - **Magnitude**: $|X[k, m]|$ — amplitude of each frequency component per frame.
    - **Power**: $|X[k, m]|^2$ — energy; emphasises dominant components.

    ### 2. Decibel scale
    $$\\text{dB}[k, m] = 20 \\log_{10} \\frac{|X[k, m]|}{\\text{ref}}$$
    Compresses the dynamic range so that both loud and quiet components are visible.
    Human hearing is logarithmic, so dB maps better to perceived loudness.

    ### 3. Mel scale
    The Mel scale warps the frequency axis to match human auditory perception.
    Low frequencies are spread out; high frequencies are compressed.
    A **Mel filterbank** applies $n_{\\text{mels}}$ triangular filters to the power spectrum.
    This is the basis of Mel-Frequency Cepstral Coefficients (MFCCs), the most
    widely used feature in speech and music recognition.

    $$\\text{mel}(f) = 2595 \\cdot \\log_{10}\\!\\left(1 + \\frac{f}{700}\\right)$$
    """
)

st.divider()

#Band energy concept
st.markdown(
    r"""
    ## Spectral band energy features

    To numerically compare recordings, the spectrum is divided into **subbands**.
    For each band $B = \{k_i, k_{i+1}, \ldots\}$:

    $$E_B[m] = \sum_{k \in B} |X[k, m]|^2 \quad \text{(band energy per frame)}$$

    $$\tilde{E}_B[m] = \frac{E_B[m]}{\sum_k |X[k, m]|^2} \quad \text{(relative band energy per frame)}$$

    $$\bar{E}_B = \frac{1}{M} \sum_{m=1}^{M} \tilde{E}_B[m] \quad \text{(time-averaged relative band energy)}$$

    $\bar{E}_B$ gives a single number per band per recording — a compact feature for
    comparing musical style, genre, or instrumentation.
    """
)

st.divider()
st.caption("DSP Exercise 3 * FH Joanneum * 2026")