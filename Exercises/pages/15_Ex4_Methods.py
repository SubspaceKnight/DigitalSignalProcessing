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
the EEG signal. The aim is to compare FIR and IIR low-pass filters, investigate
different filter orders, and analyze the influence of transition width and
filter transients.
""")

st.subheader("Data")

st.markdown("""
The data used in this exercise consist of an averaged EEG signal from channel
FCz. The signal contains the average of 565 error-related EEG epochs from
participant S05-1.

The EEG data were provided in an Excel file with two relevant columns:

| Column | Meaning |
|---|---|
| `Time` | Time axis in seconds |
| `Avg_EEG` | Averaged EEG amplitude at channel FCz |

The signal was sampled with a sampling frequency of:
""")

st.latex(r"""
f_s = 512 \, \text{Hz}
""")

st.markdown("""
The time axis ranges approximately from -8 seconds to +2 seconds around the
error onset. The column `Avg_EEG` was used as the signal to which the filters
were applied.
""")

st.subheader("FIR low-pass filter design")

st.markdown("""
First, FIR low-pass filters were designed using `scipy.signal.firwin`.

FIR means **Finite Impulse Response**. An FIR filter has an impulse response
with a finite number of non-zero samples. The filter coefficients returned by
`firwin` are called **taps**.

For FIR filters, these taps directly correspond to the impulse response:
""")

st.latex(r"""
h[n] = b[n]
""")

st.markdown("""
This is because FIR filters do not use feedback. The output only depends on
the current and previous input samples:
""")

st.latex(r"""
y[n] = \sum_{k=0}^{M} b_k x[n-k]
""")

st.markdown("""
The number of taps determines the length of the FIR filter. The FIR filter
order is the number of taps minus one:
""")

st.latex(r"""
\text{FIR filter order} = N_{\text{taps}} - 1
""")

st.markdown("""
In this exercise, FIR low-pass filters were designed with a cutoff frequency of:
""")

st.latex(r"""
f_c = 10 \, \text{Hz}
""")

st.markdown("""
The number of taps was varied in order to investigate how the filter order
affects the magnitude response, phase response, filtered signal, and filter
transient.
""")

st.subheader("IIR Butterworth low-pass filter design")

st.markdown("""
In addition to FIR filters, IIR Butterworth low-pass filters were designed
using `scipy.signal.butter`.

IIR means **Infinite Impulse Response**. In contrast to FIR filters, IIR filters
use feedback. Therefore, previous output values are used to calculate the
current output value.
""")

st.latex(r"""
y[n] =
\sum_{k=0}^{M} b_k x[n-k]
-
\sum_{k=1}^{N} a_k y[n-k]
""")

st.markdown("""
The coefficients $b_k$ describe the feedforward part of the filter, while the
coefficients $a_k$ describe the feedback part.

The Butterworth filter was chosen because it has a smooth magnitude response
in the passband. Different IIR filter orders were tested. Compared with FIR
filters, IIR filters can often achieve sharp filtering with a lower order.
However, they may introduce nonlinear phase behavior and their stability must
be considered.
""")

st.subheader("Transfer function coefficients and second-order sections")

st.markdown("""
Digital filters can be represented in different forms.

One common representation uses transfer function coefficients. In this form,
the filter is described by numerator coefficients `b` and denominator
coefficients `a`.
""")

st.latex(r"""
H(z) = \frac{B(z)}{A(z)}
""")

st.markdown("""
Here, `b` represents the feedforward part of the filter and `a` represents the
feedback part.

For FIR filters, the denominator is simply:
""")

st.latex(r"""
a = [1]
""")

st.markdown("""
For IIR filters, the transfer function representation can become numerically
problematic, especially for higher filter orders. Therefore, IIR filters can
also be represented using **second-order sections**.

A second-order section represents the filter as a cascade of smaller second
order filters. Each row of the SOS matrix has the form:
""")

st.latex(r"""
[b_0, b_1, b_2, a_0, a_1, a_2]
""")

st.markdown("""
Using second-order sections is usually more numerically stable than applying
one high-order IIR filter directly with transfer function coefficients.
Therefore, `sosfilt` is often preferred for IIR filtering.
""")

st.subheader("Magnitude and phase response")

st.markdown("""
For each designed filter, the frequency response was analyzed.

For filters represented by transfer function coefficients, the frequency
response was calculated using `scipy.signal.freqz`. For filters represented as
second-order sections, `scipy.signal.sosfreqz` was used.

The complex frequency response can be separated into magnitude and phase:
""")

st.latex(r"""
H[k] = |H[k]| \cdot e^{i \angle H[k]}
""")

st.markdown("""
The **magnitude response** describes how strongly each frequency component is
attenuated or passed by the filter.

The **phase response** describes how the filter shifts frequency components in
time.

This comparison is important because two filters may have a similar magnitude
response but a very different phase response. This can lead to different
effects in the filtered EEG waveform.
""")

st.subheader("Applying the filters")

st.markdown("""
The designed filters were applied to the EEG signal using two different
filtering functions.

For FIR filters and IIR filters represented by transfer function coefficients,
`scipy.signal.lfilter` was used.

For IIR filters represented as second-order sections, `scipy.signal.sosfilt`
was used.

The filtered signals were then compared visually in the time domain. This was
done to investigate how the different filter designs affect the EEG waveform.
Special attention was paid to smoothing effects, delay, changes in waveform
shape, and transient behavior at the beginning of the filtered signal.
""")

st.subheader("Kaiser-window FIR filter design")

st.markdown("""
In another step, FIR filter orders were estimated automatically using
`scipy.signal.kaiserord`.

The Kaiser-window method estimates the required number of taps based on a
desired attenuation and transition width.

The **transition width** describes the frequency range in which the filter
changes from passband to stopband. A small transition width means that the
filter should separate passed and attenuated frequencies very sharply.
""")

st.markdown("""
Different transition widths were tested, for example:
""")

st.latex(r"""
5 \, \text{Hz}, \quad 2 \, \text{Hz}, \quad 1 \, \text{Hz}, \quad 0.2 \, \text{Hz}
""")

st.markdown("""
Smaller transition widths require higher filter orders. This means that the
number of taps increases. As a result, the filter can become very long, which
may lead to stronger filter transients and larger delays.

This part of the analysis was used to show that automatic filter order
estimation can produce impractically high filter orders if the transition width
is chosen too small.
""")

st.subheader("Full-signal filtering versus segment filtering")

st.markdown("""
The EEG signal was filtered in two different ways.

First, the complete signal was filtered and the segment of interest was
selected afterwards.

Second, the segment of interest was selected first and only this shorter
segment was filtered.

This comparison is important because filters require previous samples to
produce reliable output values. At the beginning of a signal or segment, these
previous samples are missing. This can cause an artificial distortion called a
**filter transient**.
""")

st.markdown("""
A filter transient is not part of the original EEG signal. It is caused by the
filtering process itself. The effect becomes stronger for higher-order filters
or filters with many taps.

Therefore, filtering the full signal first and segmenting afterwards is usually
the safer approach, especially when short signal segments are analyzed.
""")

st.subheader("Method pipeline")

st.markdown("""
The complete analysis followed this workflow:

1. The EEG signal was loaded from the Excel file.
2. The sampling frequency and cutoff frequency were defined.
3. FIR low-pass filters with different numbers of taps were designed.
4. IIR Butterworth low-pass filters with different orders were designed.
5. Magnitude and phase responses were calculated.
6. The filters were applied to the EEG signal.
7. Kaiser-window FIR filters were designed for different transition widths.
8. Filtering the full signal was compared with filtering only a segment.
9. The results were interpreted with respect to filter order, transition width,
   phase behavior and filter transients.
""")