import streamlit as st
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Ex3 - Discussion", layout="wide")
st.title("Discussion - STFT Results & Interpretation")

#TODO Task: STFT differences across recordings
st.markdown(
    """
    ## Differences across recordings

    <!-- TODO: fill in after analysis. Template questions to answer:
         - Which recording shows more high-frequency energy? Why?
         - Which shows more temporal variation in spectral content? Why?
         - Are there clear tonal components visible as horizontal lines?
           If so, at which frequencies?
         - Are there percussive transients visible as vertical bright stripes?
    -->

    *TODO — describe noticeable differences in the STFT results between Song A and Song B.
    Reference specific observations from the spectrogram (e.g. "Song A shows a sustained
    energy band at 440 Hz corresponding to …").*
    """
)

st.divider()

#Y-axis representation comparison
st.markdown(
    """
    ## Y-axis representation — what each reveals

    ### Magnitude
    Direct amplitude of each frequency component. Easy to interpret but the
    dynamic range is often too large — quiet components disappear next to loud ones.

    ### Power
    Squares the magnitude, further emphasising dominant components. Useful when
    the energy distribution (not amplitude) is the quantity of interest.

    ### Decibel scale
    Applies a $20 \\log_{10}$ compression. Maps best to human loudness perception
    because our ears respond logarithmically. Reveals quiet structures (e.g. reverb
    tails, breath noise) that are invisible on linear scales.
    **Most useful representation for general music analysis.**

    ### Mel scale
    Warps the frequency axis to match critical-band spacing in the human auditory system.
    Low frequencies get more resolution; high frequencies are compressed.
    Removes redundant spectral detail above ~4 kHz, making patterns in the perceptually
    important range more visible. Standard input for ML audio classifiers.

    <!-- TODO: state which representation was most useful for *our* songs and why. -->
    """
)

st.divider()

#TODO: Time-frequency trade-off
st.markdown(
    r"""
    ## The time-frequency trade-off

    The window length $L$ governs a fundamental trade-off that cannot be avoided —
    it is a consequence of the Heisenberg-Gabor uncertainty principle for discrete signals:

    $$\Delta f \cdot \Delta t \geq \frac{1}{2}$$

    In practice:

    | Setting | $\Delta f = f_s / L$ | $\Delta t = H / f_s$ | Best for |
    |---|---|---|---|
    | Long window (large $L$) | Small (fine freq.) | Large (coarse time) | Harmonic analysis, pitch detection |
    | Short window (small $L$) | Large (coarse freq.) | Small (fine time) | Onset detection, transients, drums |

    **For music:** a window of 1024-2048 samples at 44 100 Hz (≈23-46 ms) is a common
    starting point. Percussive content benefits from shorter windows; sustained harmonic
    content from longer ones. Some applications use **multi-resolution STFTs** that
    run several window sizes simultaneously and combine the results.

    <!-- TODO: state what window length qw settled on and why it suited our recordings. -->
    """
)

st.divider()

#TODO:Window function effects
st.markdown(
    """
    ## Effect of window functions

    The Hann window is the standard choice for audio: it suppresses spectral leakage
    to -31 dB side-lobe level while keeping the main lobe reasonably narrow.

    - **Rectangular** — narrowest main lobe but -13 dB side-lobes. Nearby frequency
      components bleed into each other, making pitch discrimination unreliable.
    - **Hamming** — similar main lobe width to Hann but with a slightly higher minimum
      side-lobe level (good for speech, standard in MFCCs).
    - **Blackman** — widest main lobe but -57 dB side-lobes. Worth the cost when the
      dynamic range between components is very large (e.g. a faint harmonic next to
      a loud fundamental).

    <!-- TODO: describe what we observed in the spectrograms when switching windows. -->
    """
)

st.divider()

#TODO: Band energy interpretation
st.markdown(
    """
    ## Band energy results — interpretation

    <!-- TODO: fill in with our specific numbers once computed. Template: -->

    | Band | Song A $\\bar{E}_B$ | Song B $\\bar{E}_B$ | Interpretation |
    |---|---|---|---|
    | 0-250 Hz | TODO | TODO | Sub-bass / kick drum |
    | 250-2000 Hz | TODO | TODO | Vocal / melody range |
    | 2000-8000 Hz | TODO | TODO | Presence / harmonics |
    | 8000-Nyquist Hz | TODO | TODO | Air / cymbals |

    *TODO — which song is more bass-heavy? Which has more high-frequency content?
    Does this match our subjective impression of the recordings?*
    """
)

st.divider()

#TODO: Spectral centroid and flatness
st.markdown(
    """
    ## Spectral centroid — interpretation

    <!-- TODO: insert mean centroid values per song and discuss. -->

    The spectral centroid over time reveals how the timbral brightness evolves.
    Transient events (drum hits, consonants) produce brief centroid spikes.
    Sustained sections with heavy bass pull the centroid down.

    *TODO — compare mean centroids between songs. Does the centroid correlate
    with our perception of brightness or genre?*

    ## Spectral flatness — interpretation

    *TODO — compare flatness between songs. Does one recording have more noise-like
    frames (high flatness) vs tonal frames (low flatness)?
    This could indicate more percussive vs more harmonic content.*
    """
)

st.divider()

#TODO: Feature choice for ML classification
st.markdown(
    """
    ## If we had to train a classifier

    If the task were to train a model that distinguishes the recordings into
    classes (genre, mood, artist, etc.), the following features would be
    strong candidates:

    | Feature | Dimensionality | Why useful |
    |---|---|---|
    | MFCCs (13-40 coefficients) | Per frame | Compact perceptual representation; industry standard |
    | $\\bar{E}_B$ per band | 3-4 scalars per recording | Captures energy distribution across the spectrum |
    | Mean spectral centroid | 1 scalar | Single-number brightness descriptor |
    | Std of spectral centroid | 1 scalar | Captures dynamic range of brightness |
    | Spectral flatness (mean) | 1 scalar | Tonal vs percussive tendency |
    | Tempo / beat period | 1 scalar | Rhythmic signature (requires onset detection) |
    | Zero-crossing rate | 1 scalar per frame | Simple noisiness / voiced-unvoiced proxy |

    For a classical ML pipeline (SVM, random forest), the **time-averaged** versions
    of all per-frame features (mean + std) form a compact fixed-length feature vector.
    For a deep learning pipeline (CNN, transformer), the raw **Mel spectrogram** is
    typically fed directly as a 2D image.

    <!-- TODO: state which features we would prioritise for our specific pair of
         recordings and give a brief justification. -->
    """
)

st.divider()
st.caption("DSP Exercise 3 * FH Joanneum * 2026")