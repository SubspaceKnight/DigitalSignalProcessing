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
import Exercises.ex3.analysis as analysis

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

    The STFT divides the signal into short, overlapping frames and applies the DFT/FFT to each:

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

#TODO: add a simple plot of a two-tone signal and its STFT spectrogram with different window lengths to visually demonstrate the trade-off.
#Demo: two sinusoids with different window lengths
st.markdown("### Interactive demo — effect of window length on a two-tone signal")

fs_demo = 8000
t_demo = np.linspace(0, 1.0, fs_demo, endpoint=False)
#Two tones: 440 Hz and 880 Hz
x_demo  = np.sin(2 * np.pi * 440 * t_demo) + 0.7 * np.sin(2 * np.pi * 880 * t_demo)

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

fig_demo = go.Figure(go.Heatmap(
    x=result_demo["times"],
    y=result_demo["freqs"],
    z=mag_db_demo,
    colorscale="Magma",
    zmin=mag_db_demo.max() - 60,
    zmax=mag_db_demo.max(),
    colorbar=dict(title="dB"),
))
fig_demo.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Frequency (Hz)",
    yaxis_range=[0, 2000],
    title=(
        f"STFT spectrogram — window = {win_len_demo} samples "
        f"({win_len_demo / fs_demo * 1000:.1f} ms) · "
        f"Δf = {fs_demo / win_len_demo:.1f} Hz · "
        f"window = {win_fn_demo}"
    ),
    height=340,
    margin=dict(l=60, r=20, t=50, b=60),
)
st.plotly_chart(fig_demo, use_container_width=True)
st.caption(
    "A 440 Hz + 880 Hz two-tone signal. "
    "Increase the window length to resolve the two frequency lines more sharply. "
    "Decrease it to gain time resolution at the cost of frequency precision."
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
colors = ["cornflowerblue", "coral", "mediumseagreen", "goldenrod"] #I wanna make them look nice and distinct, but feel free to change:) maybe use a color palette library like seaborn or plotly.express for more options
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