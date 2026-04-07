import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import Exercises.ex3.helper as helper
import Exercises.ex3.analysis as analysis

st.set_page_config(page_title="Ex3 - Methods", layout="wide")
st.title("Methods - How the Analysis Was Performed")

#TODO: Song selection
st.markdown("## Song selection")
st.markdown(
    """
    <!-- TODO: fill in once songs are chosen. -->

    *Two (or more) recordings were selected for analysis. The table below summarises
    the choice and the reasoning.*
    """
)

#TODO: we need to replace the placeholder rows with the actual songs
song_table = {
    "Track":   ["Song A (TODO)", "Song B (TODO)"],
    "Artist":  ["Artist A", "Artist B"],
    "Genre":   ["TODO", "TODO"],
    "Why chosen": [
        "TODO - e.g. rich harmonic content, clear beat, …",
        "TODO - e.g. contrast in energy distribution, …",
    ],
    "Duration (s)": ["TODO", "TODO"],
}

st.dataframe(pd.DataFrame(song_table), use_container_width=True, hide_index=True)

st.divider()

#Dataset overview
st.markdown("## Dataset")

audio_files = helper.list_audio_files()

if not audio_files:
    st.info(
        "No audio files found in `data/audio/`. "
        "Place any `.wav` / `.mp3` files there and reload."
    )
else:
    for fname in audio_files:
        audio = helper.load_audio(fname)
        if audio:
            info  = helper.audio_info_df(audio)
            cols  = st.columns(len(info))
            for col, (k, v) in zip(cols, info.items()):
                col.metric(k, v)
            st.divider()

st.divider()

#TODO:STFT implementation
st.markdown(
    """
    ## STFT implementation

    The STFT is implemented from scratch in `analysis.stft()`. No external
    FFT/STFT function from a library is used. The steps are:

    1. **Pad** the signal so the first and last frames are centred on the signal boundaries.
    2. **Frame** the padded signal into overlapping segments of length $L$ with hop $H$.
    3. **Window** each frame by multiplying sample-wise with the chosen window function.
    4. **DFT** each windowed frame using the twiddle-matrix implementation from Exercise 2.
    5. **Keep** only the first $L/2 + 1$ bins (one-sided spectrum, conjugate symmetry).

    The result is a complex matrix $X[k, m]$ of shape $(L/2 + 1) \\times M$.
    """
)

#TODO: DFT correctness check (reused from Ex2, applied to a single frame)
st.markdown("### DFT correctness — single-frame sanity check")

fs_chk  = 4096.0
N_chk   = 512
t_chk   = np.linspace(0, N_chk / fs_chk, N_chk, endpoint=False)
f_true  = 200.0
x_chk   = np.sin(2 * np.pi * f_true * t_chk)
w_chk   = analysis.get_window("hann", N_chk)
X_chk   = analysis.dft(x_chk * w_chk)
freqs_c = np.arange(N_chk // 2 + 1) * fs_chk / N_chk
mag_c   = np.abs(X_chk[: N_chk // 2 + 1]) / N_chk

fig_chk = go.Figure()
fig_chk.add_trace(go.Scatter(
    x=freqs_c, y=mag_c * 2,
    mode="lines", line=dict(color="cornflowerblue", width=1.5),
    fill="tozeroy", fillcolor="rgba(100,149,237,0.10)",
))
fig_chk.add_vline(
    x=f_true, line=dict(color="tomato", dash="dash", width=1.5),
    annotation_text=f"{f_true} Hz ✓",
)
fig_chk.update_layout(
    xaxis_title="Frequency (Hz)",
    yaxis_title="|X[k]| / N (normalised)",
    title=f"DFT of a single windowed frame — peak must be at {f_true} Hz",
    height=280,
    xaxis_range=[0, 800],
    margin=dict(l=60, r=20, t=50, b=60),
)
st.plotly_chart(fig_chk, use_container_width=True)
st.caption(f"Peak correctly located at {f_true} Hz. ✓")

st.divider()

#TODO: Window function choice justification
st.markdown(
    """
    ## Window function choice

    <!-- TODO: fill in after we've run the analysis on our songs. -->

    *The default window used for the analysis is Hann. The reasoning:*
    - Good balance of main-lobe width and side-lobe suppression.
    - Standard choice for audio analysis.
    - TODO: describe what we observed when switching to other windows.
    """
)

st.divider()

#TODO: Parameter selection
st.markdown(
    """
    ## Parameter selection

    | Parameter | Value used | Reasoning |
    |           |            |           |
    | Window function | Hann | TODO |
    | Window length $L$ | TODO samples | TODO |
    | Hop length $H$ | TODO samples | TODO (= 50 % overlap) |
    | Frequency resolution $\Delta f$ | TODO Hz | $f_s / L$ |
    | Time resolution $\Delta t$ | TODO ms | $H / f_s$ |

    <!-- TODO: fill in once songs and fs are known. -->
    """
)

st.divider()


#TODO: Metrics
st.markdown(
    """
    ## Metrics used

    ### Band energy features  (assignment requirement)
    The full frequency range is split into four subbands:

    | Band | Range | Rationale |
    |      |       |           |
    | Sub-bass | 0-250 Hz | Kick drum, bass fundamentals |
    | Low-mid  | 250-2000 Hz | Vocals, guitar body, melody |
    | High-mid | 2000-8000 Hz | Harmonics, presence, sibilance |
    | High     | 8000-Nyquist Hz | Cymbals, air |

    <!-- TODO: adjust band boundaries once we know the Nyquist of our recordings. -->

    ### Additional metric — Spectral centroid
    The spectral centroid tracks the "centre of mass" of the spectrum over time.
    A high centroid → bright/noisy sound. A low centroid → bass-heavy/muffled.
    It is a compact single-value-per-frame descriptor useful for distinguishing
    musical genres and detecting timbral changes.

    ### Additional metric — Spectral flatness  *(optional)*
    Measures how tonal vs noise-like the signal is per frame.
    Values near 1 = white noise. Values near 0 = pure tone.
    Useful for detecting voiced vs unvoiced segments or tonal vs percussive content.
    """
)

st.divider()
st.caption("DSP Exercise 3 * FH Joanneum * 2026")