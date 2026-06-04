import streamlit as st

# page config
st.set_page_config(
    page_title="Exercise 4 - Methods",
    layout="wide"
)

st.title("Exercise 4 - Digital Filtering")
st.header("2. Methods")

st.markdown("""
This section describes the methods used to design and apply digital filters to
the EEG signal. The main goal was to compare FIR and IIR low-pass filters and
to investigate the influence of filter order, transition width, and filter
transients.
""")

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
st.subheader("Data")

st.markdown("""
The data consist of an averaged EEG signal from channel FCz. The signal contains
the average of 565 error-related EEG epochs from participant S05-1.

The Excel file contains two relevant columns:

| Column | Meaning |
|---|---|
| `Time` | Time axis in seconds |
| `Avg_EEG` | Averaged EEG amplitude at channel FCz |

The sampling frequency was:
""")

st.latex(r"""
f_s = 512 \, \text{Hz}
""")

st.markdown("""
The signal ranges approximately from -8 seconds to +2 seconds around the error
onset. The column `Avg_EEG` was used as the signal for filtering.
""")

# ------------------------------------------------------------
# FIR filter design
# ------------------------------------------------------------
st.subheader("FIR low-pass filter design")

st.markdown("""
FIR low-pass filters were designed using `scipy.signal.firwin`.

FIR means **Finite Impulse Response**. The coefficients returned by `firwin`
are called **taps**. For FIR filters, these coefficients directly correspond
to the impulse response:
""")

st.latex(r"""
h[n] = b[n]
""")

st.markdown("""
This is the case because FIR filters do not use feedback. The output only
depends on the current and previous input samples. Therefore, the filter is
fully described by its finite set of coefficients.
""")

st.latex(r"""
y[n] = \sum_{k=0}^{M} b_k x[n-k]
""")

st.markdown("""
The number of taps determines the filter length. The FIR filter order is:
""")

st.latex(r"""
\text{FIR filter order} = N_{\text{taps}} - 1
""")

st.markdown("""
All FIR filters were designed as low-pass filters with a cutoff frequency of:
""")

st.latex(r"""
f_c = 10 \, \text{Hz}
""")

st.markdown("""
The number of taps was varied to investigate how the FIR filter order influences
the magnitude response, phase response, filtered signal, and filter transient.
""")

# ------------------------------------------------------------
# IIR filter design
# ------------------------------------------------------------
st.subheader("IIR Butterworth low-pass filter design")

st.markdown("""
IIR Butterworth low-pass filters were designed using `scipy.signal.butter`.

IIR means **Infinite Impulse Response**. In contrast to FIR filters, IIR filters
use feedback. Therefore, previous output values can influence the current output
value.
""")

st.latex(r"""
y[n] =
\sum_{k=0}^{M} b_k x[n-k]
-
\sum_{k=1}^{N} a_k y[n-k]
""")

st.markdown("""
The coefficients \(b_k\) describe the feedforward part of the filter, while the
coefficients \(a_k\) describe the feedback part.

Different IIR filter orders were tested. Compared with FIR filters, IIR filters
can often achieve sharp filtering with lower order, but their phase response is
usually nonlinear and stability has to be considered.
""")

# ------------------------------------------------------------
# Filter representation
# ------------------------------------------------------------
st.subheader("Filter representation")

st.markdown("""
Digital filters can be represented by transfer function coefficients. In this
form, the filter is described by numerator coefficients `b` and denominator
coefficients `a`.
""")

st.latex(r"""
H(z) = \frac{B(z)}{A(z)}
""")

st.markdown("""
Here, `b` describes the feedforward part and `a` describes the feedback part.

For FIR filters, there is no feedback part, so the denominator is simply:
""")

st.latex(r"""
a = [1]
""")

st.markdown("""
For IIR filters, the transfer function representation with coefficients `b`
and `a` can become numerically problematic, especially for higher filter orders.

Therefore, IIR filters can also be represented as **second-order sections**
(`sos`). The SOS matrix splits a higher-order filter into several smaller
second-order filters. This is usually more numerically stable and is therefore
preferred for higher-order IIR filtering.
""")

# ------------------------------------------------------------
# Frequency response
# ------------------------------------------------------------
st.subheader("Magnitude and phase response")

st.markdown("""
For each designed filter, the frequency response was calculated.

Filters represented by transfer function coefficients were analyzed using
`scipy.signal.freqz`. Filters represented as second-order sections were analyzed
using `scipy.signal.sosfreqz`.

The frequency response was evaluated in terms of magnitude and phase:

- the **magnitude response** shows which frequencies are passed or attenuated
- the **phase response** shows how frequency components are shifted in time
""")

# ------------------------------------------------------------
# Applying filters
# ------------------------------------------------------------
st.subheader("Applying the filters")

st.markdown("""
The designed filters were applied to the EEG signal.

For FIR filters and IIR filters represented by transfer function coefficients,
`scipy.signal.lfilter` was used. For FIR filters, this is straightforward
because the FIR coefficients are the impulse response of the filter.

For IIR filters represented as second-order sections, `scipy.signal.sosfilt`
was used. This was used to compare transfer function filtering with SOS-based
filtering.

The filtered signals were compared visually in the time domain to observe
smoothing, delay, changes in waveform shape, and transient effects.
""")

# ------------------------------------------------------------
# Kaiser-window FIR design
# ------------------------------------------------------------
st.subheader("Kaiser-window FIR filter design")

st.markdown("""
In another step, FIR filter orders were estimated automatically using
`scipy.signal.kaiserord`.

The Kaiser-window method estimates the required number of taps based on the
desired attenuation and the transition width. The transition width describes
how quickly the filter changes from passband to stopband.

The following transition widths were tested:
""")

st.latex(r"""
5 \, \text{Hz}, \quad 2 \, \text{Hz}, \quad 1 \, \text{Hz}, \quad 0.2 \, \text{Hz}
""")

st.markdown("""
For each estimated number of taps, a new FIR low-pass filter with
`scipy.signal.firwin` was designed and then applied to the EEG signal with
`scipy.signal.lfilter`.

Smaller transition widths require higher filter orders. This can lead to very
long filters, stronger filter transients, and larger delays.
""")

# ------------------------------------------------------------
# Full signal versus segment filtering
# ------------------------------------------------------------
st.subheader("Full-signal filtering versus segment filtering")

st.markdown("""
The Kaiser-window FIR filters were applied in two different ways.

First, the full 10-second signal was filtered and the last 2 seconds were
selected afterwards.

Second, the last 2 seconds were selected first and only this shorter segment was
filtered.

This comparison was used to investigate **filter transients**. A filter
transient is an artificial distortion at the beginning of a filtered signal or
segment. It occurs because the filter does not yet have enough previous samples
available.

Filtering the full signal first and segmenting afterwards is usually the safer
approach for short signal segments, because the filter has more previous samples
available before the segment of interest begins.
""")

# ------------------------------------------------------------
# Method pipeline
# ------------------------------------------------------------
st.subheader("Method pipeline")

st.markdown("""
The analysis followed this workflow:

1. Load the EEG signal from the Excel file.
2. Define the sampling frequency and cutoff frequency.
3. Design FIR low-pass filters with different numbers of taps using `firwin`.
4. Design IIR Butterworth low-pass filters with different orders using `butter`.
5. Calculate magnitude and phase responses.
6. Apply FIR and IIR filters using `lfilter`.
7. Apply IIR filters using `sosfilt` for the SOS representation.
8. Estimate FIR orders with `kaiserord` for different transition widths.
9. Apply Kaiser-window FIR filters to the full signal and to the last 2 seconds.
10. Compare full-signal filtering with segment-only filtering.
""")