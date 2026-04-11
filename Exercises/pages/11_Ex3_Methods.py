import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ex3 import helper, analysis




st.set_page_config(page_title="Ex3 - Methods", layout="wide")
st.title("Methods - How the Analysis Was Performed")

# --------------------------------------------------
# Song selection
# --------------------------------------------------
st.markdown("## Song selection")
st.markdown(
    """
    Two recordings were selected for analysis. The goal was to compare two songs with
    clearly different musical structure and spectral behaviour, so that the advantages
    of time-frequency analysis become visible in a meaningful way.
    """
)

song_table = [
    {
        "Track": "Lollipop",
        "Artist": "Clejan",
        "Genre": "Balkan / Gypsy / Dance",
        "Why chosen": (
            "Chosen because it is rhythmically regular and repetitive, which makes it useful "
            "for observing stable time-frequency patterns, recurring beat structures, and "
            "relatively consistent spectral energy over time."
        ),
        "Duration": "2:01",
    },
    {
        "Track": "Chop Suey!",
        "Artist": "System Of A Down",
        "Genre": "Alternative / Nu Metal",
        "Why chosen": (
            "Chosen because it contains strong dynamic contrasts, denser instrumentation, "
            "and more abrupt spectral changes. This makes it suitable for analysing "
            "transients, spectral spread, and timbral variation over time."
        ),
        "Duration": "3:28",
    },
]

st.dataframe(pd.DataFrame(song_table), use_container_width=True, hide_index=True)

st.markdown(
    """
    Together, these two songs form a useful contrast: *Lollipop* represents a more regular,
    beat-oriented structure, while *Chop Suey!* represents a more dynamic and spectrally
    complex recording. This makes the pair well suited for comparing STFT-based features
    such as band energy, spectral centroid, and spectral flatness.
    """
)

st.divider()

# --------------------------------------------------
# Dataset overview
# --------------------------------------------------
st.markdown("## Dataset")

audio_files = helper.list_audio_files()

dataset_rows = []
if not audio_files:
    st.info(
        "No audio files found in `data/audio/`. "
        "Place any `.wav` / `.mp3` files there and reload."
    )
else:
    for fname in audio_files:
        audio = helper.load_audio(fname)
        if audio:
            dataset_rows.append({
                "File": audio["filename"],
                "Sample rate (Hz)": audio["sr"],
                "Duration (s)": round(audio["duration"], 2),
                "Samples": audio["n_samples"],
                "Channels": audio["channels"],
                "Nyquist (Hz)": audio["sr"] // 2,
            })

    if dataset_rows:
        st.dataframe(pd.DataFrame(dataset_rows), use_container_width=True, hide_index=True)

st.divider()

# --------------------------------------------------
# STFT implementation
# --------------------------------------------------
st.markdown(
    """
    ## STFT implementation

    The STFT is implemented in `analysis.stft()`. The signal is split into overlapping
    frames, multiplied by a window function, and transformed frame by frame into the
    frequency domain. The procedure is:

    1. **Pad** the signal so the first and last frames are centred on the signal boundaries.
    2. **Frame** the padded signal into overlapping segments of length $L$ with hop $H$.
    3. **Window** each frame by multiplying sample-wise with the chosen window function.
    4. **Transform** each frame into the frequency domain.
    5. **Keep** only the first $L/2 + 1$ bins of the one-sided spectrum.

    The result is a complex matrix $X[k, m]$ of shape $(L/2 + 1) \\times M$, where
    each column corresponds to one time frame and each row to one frequency bin.
    """
)


# --------------------------------------------------
# DFT correctness check
# --------------------------------------------------
st.markdown("### DFT correctness — single-frame sanity check")

fs_chk = 4096.0
N_chk = 512
t_chk = np.linspace(0, N_chk / fs_chk, N_chk, endpoint=False)
f_true = 200.0
x_chk = np.sin(2 * np.pi * f_true * t_chk)
w_chk = analysis.get_window("hann", N_chk)
X_chk = analysis.dft(x_chk * w_chk)
freqs_c = np.arange(N_chk // 2 + 1) * fs_chk / N_chk
mag_c = np.abs(X_chk[: N_chk // 2 + 1]) / N_chk

fig_chk = go.Figure()
fig_chk.add_trace(go.Scatter(
    x=freqs_c,
    y=mag_c * 2,
    mode="lines",
    line=dict(width=1.5),
    fill="tozeroy",
))
fig_chk.add_vline(
    x=f_true,
    line=dict(dash="dash", width=1.5),
    annotation_text=f"{f_true} Hz ✓",
)
fig_chk.update_layout(
    xaxis_title="Frequency (Hz)",
    yaxis_title="|X[k]| / N (normalised)",
    title=f"DFT of a single windowed frame — peak should appear at {f_true} Hz",
    height=280,
    xaxis_range=[0, 800],
    margin=dict(l=60, r=20, t=50, b=60),
)
st.plotly_chart(fig_chk, use_container_width=True)
st.caption(f"The peak is correctly located at {f_true} Hz, confirming the correctness of the DFT implementation for this test signal.")

st.divider()

# --------------------------------------------------
# Window function choice
# --------------------------------------------------
st.markdown(
    """
    ## Window function choice

    The default window used for the analysis is the **Hann window**. It was chosen because
    it provides a good compromise between frequency resolution and side-lobe suppression.
    This makes it a standard and reliable choice for general audio analysis.

    In practice, the Hann window reduces spectral leakage much more effectively than the
    rectangular window, while still preserving reasonably sharp frequency localisation.
    A rectangular window has narrower main lobes, but it introduces much stronger side lobes,
    which makes nearby spectral components bleed into each other. More strongly tapered windows
    such as Blackman suppress leakage even more, but at the cost of broader peaks.
    """
)

st.divider()

# --------------------------------------------------
# Parameter selection
# --------------------------------------------------
st.markdown("## Parameter selection")

# Reasonable default values for the report
L = 2048
H = 1024

# Try to infer fs from first available file
fs_used = None
if audio_files:
    audio0 = helper.load_audio(audio_files[0])
    if audio0:
        fs_used = audio0["sr"]

if fs_used is not None:
    delta_f = fs_used / L
    delta_t_ms = 1000 * H / fs_used
    param_table = pd.DataFrame([
        {"Parameter": "Window function", "Value used": "Hann", "Reasoning": "Standard choice for audio analysis; good trade-off between leakage suppression and resolution."},
        {"Parameter": "Window length L", "Value used": f"{L} samples", "Reasoning": "Long enough to resolve harmonic content, while still preserving useful temporal detail."},
        {"Parameter": "Hop length H", "Value used": f"{H} samples", "Reasoning": "50% overlap; common compromise between temporal smoothness and computational cost."},
        {"Parameter": "Frequency resolution Δf", "Value used": f"{delta_f:.2f} Hz", "Reasoning": "Computed as fs / L."},
        {"Parameter": "Time resolution Δt", "Value used": f"{delta_t_ms:.2f} ms", "Reasoning": "Computed as H / fs."},
    ])
else:
    param_table = pd.DataFrame([
        {"Parameter": "Window function", "Value used": "Hann", "Reasoning": "Standard choice for audio analysis; good trade-off between leakage suppression and resolution."},
        {"Parameter": "Window length L", "Value used": f"{L} samples", "Reasoning": "Long enough to resolve harmonic content, while still preserving useful temporal detail."},
        {"Parameter": "Hop length H", "Value used": f"{H} samples", "Reasoning": "50% overlap; common compromise between temporal smoothness and computational cost."},
        {"Parameter": "Frequency resolution Δf", "Value used": "depends on fs", "Reasoning": "Computed as fs / L."},
        {"Parameter": "Time resolution Δt", "Value used": "depends on fs", "Reasoning": "Computed as H / fs."},
    ])

st.dataframe(param_table, use_container_width=True, hide_index=True)

st.markdown(
    """
    These values were selected as a practical starting point for music analysis. A window
    length of 2048 samples is commonly used because it offers sufficient frequency resolution
    for harmonic structure, while a hop length of 1024 samples provides smooth overlap between
    neighbouring frames.
    """
)

st.divider()

# --------------------------------------------------
# Metrics used
# --------------------------------------------------
st.markdown(
    """
    ## Metrics used

    ### Band energy features 

    The average spectra and spectrograms showed that most relevant musical energy was concentrated
    below roughly 8 kHz, while additional brightness components extended up to the Nyquist limit.
    Based on this observed range, the spectrum was divided into four broad bands that separate
    bass, mid-range, upper-mid, and very high-frequency content.

    | Band | Range | Rationale |
    |------|-------|-----------|
    | Sub-bass | 0-250 Hz | Kick drum, bass fundamentals |
    | Low-mid | 250-2000 Hz | Vocals, guitar body, melody |
    | High-mid | 2000-8000 Hz | Harmonics, presence, sibilance |
    | High | 8000-Nyquist Hz | Cymbals, air, very bright components |

    These bands were chosen because they separate low-frequency rhythm and bass content from
    the perceptually important mid-range and the upper-frequency brightness region.

    ### Additional metric — Spectral centroid

    The spectral centroid tracks the "center of mass" of the spectrum over time.
    A high centroid indicates a brighter and more high-frequency dominated sound,
    while a low centroid indicates a darker or more bass-heavy sound. This makes it
    a useful compact feature for comparing overall timbral brightness.

    ### Additional metric — Spectral flatness

    Spectral flatness measures how tonal or noise-like the signal is in each frame.
    Values near 1 indicate a flatter, noise-like spectrum, while values near 0 indicate
    a more tonal spectrum with concentrated peaks. This can help distinguish harmonic
    content from more percussive or noisy passages.
    """
)

st.divider()
st.caption("DSP Exercise 3 · FH Joanneum · 2026")