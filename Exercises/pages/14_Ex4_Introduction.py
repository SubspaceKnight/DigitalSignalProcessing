import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# page config
st.set_page_config(
    page_title="Exercise 4 - Digital Filtering",
    layout="wide"
)

st.title("Exercise 4 - Digital Filtering")

# Short summary of the page content.
st.markdown("""
This page introduces the main concepts of **digital filtering** that are relevant
for Exercise 4. The focus is on FIR (Finite Impulse Response) and IIR (Infinite Impulse Response) 
filters, impulse responses, transfer functions, poles and zeros, filter order, phase behavior, 
and the practical consequences of filtering EEG signals.
""")

st.header("1. Introduction")

st.markdown("""
In previous parts of the course, signals were mainly analyzed in different
domains, for example in the time domain and in the frequency domain using the
DFT or STFT.

In this exercise, the goal is not only to describe a signal, but also to actively 
modify its frequency content using digital filtering.
""")

st.subheader("What is a digital filter?")

st.markdown("""
A digital filter can be understood as a system that transforms an input signal
$x[n]$ into an output signal $y[n]$:

$y[n] = H\\{x[n]\\}$

In practical signal processing, filters are often used to suppress unwanted
frequency components while allowing wanted frequency components to pass.

For example, a low-pass filter keeps slow signal components and attenuates
high-frequency components.
""")

# Simple filter system diagram
# an input signal x[n] is passed through a filter system H,
# resulting in an output signal y[n].
fig, ax = plt.subplots(figsize=(8, 2))

ax.text(0.1, 0.5, r"Input signal $x[n]$", ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))

ax.text(0.5, 0.5, r"Filter $H$", ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))

ax.text(0.9, 0.5, r"Output signal $y[n]$", ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))

ax.annotate("", xy=(0.38, 0.5), xytext=(0.22, 0.5),
            arrowprops=dict(arrowstyle="->"))

ax.annotate("", xy=(0.78, 0.5), xytext=(0.62, 0.5),
            arrowprops=dict(arrowstyle="->"))

ax.axis("off")
st.pyplot(fig)


st.subheader("Impulse response")

st.markdown("""
A filter can be characterized by its response to a unit impulse. The unit
impulse is defined as:
""")

st.latex(r"""
\delta[n] =
\begin{cases}
1, & n = 0 \\
0, & n \neq 0
\end{cases}
""")

st.markdown("""
The output of the filter to this impulse is called the **impulse response**
$h[n]$.

For linear time-invariant systems, the output signal can be written as a
convolution of the input signal with the impulse response:

$y[n] = x[n] * h[n]$

This means that the impulse response describes how the filter behaves in the
time domain.
""")


# impulse response of a moving-average filter
# A simple moving-average filter has a finite impulse response.
# Here, the filter averages three neighboring samples.
n = np.arange(-2, 8)

# Create a simple impulse response h[n] = 1/3 * [1, 1, 1]
h = np.zeros_like(n, dtype=float)
h[(n >= 0) & (n <= 2)] = 1 / 3

fig, ax = plt.subplots(figsize=(8, 3))

ax.stem(n, h)
ax.set_title("Impulse response of a simple 3-point moving-average FIR filter")
ax.set_xlabel("Sample index n")
ax.set_ylabel("Amplitude")
ax.grid(True)

st.pyplot(fig)

st.markdown("""
The graph shows a simple impulse response of a 3-point moving-average FIR filter. 
This is a simple smoothing filter. It averages three adjacent values, thereby smoothing 
the signal.
""")
st.subheader("Frequency response")

st.markdown("""
Filtering is mainly about changing frequency content. Therefore, it is useful
to look at the filter in the frequency domain.

The frequency response $H[k]$ is obtained from the impulse response $h[n]$
using the DFT:

$H[k] = DFT\\{h[n]\\}$


The frequency response tells us how strongly each frequency component is
changed by the filter.

Because $H[k]$ is complex-valued, it can be split into:

- the **magnitude response**, which describes how much each frequency is amplified or attenuated
- the **phase response**, which describes how frequency components are shifted in time
""")


st.subheader("Filter classes based on magnitude response")

st.markdown("""
Filters can be classified according to which frequency components they allow
to pass and which they attenuate.

| Filter type | Effect |
|---|---|
| Low-pass filter | Keeps low frequencies and attenuates high frequencies |
| High-pass filter | Keeps high frequencies and attenuates low frequencies |
| Band-pass filter | Keeps frequencies inside a selected frequency band |
| Band-stop / notch filter | Attenuates frequencies inside a selected frequency band |
| All-pass filter | Keeps the magnitude of all frequencies but may change the phase |

In this exercise, the focus is on **low-pass filtering**. The EEG signal is
filtered with a cutoff frequency of $f_c = 10\\,Hz$.
""")


# These plots show the idealized behavior of common filter classes.
# Real filters cannot have perfectly sharp transitions like this.
freq = np.linspace(-20, 20, 1000)

fc = 5
fc1 = 5
fc2 = 12

lowpass = np.where(np.abs(freq) <= fc, 1, 0)
highpass = np.where(np.abs(freq) >= fc, 1, 0)
bandpass = np.where((np.abs(freq) >= fc1) & (np.abs(freq) <= fc2), 1, 0)
bandstop = np.where((np.abs(freq) >= fc1) & (np.abs(freq) <= fc2), 0, 1)

filter_responses = {
    "Ideal low-pass filter": lowpass,
    "Ideal high-pass filter": highpass,
    "Ideal band-pass filter": bandpass,
    "Ideal band-stop filter": bandstop,
}

for title, response in filter_responses.items():
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(freq, response)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude response")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)

    st.pyplot(fig)


st.subheader("Ideal filters and real filters")

st.markdown("""
An ideal filter would perfectly separate the passband from the stopband.

For an ideal low-pass filter:

- frequencies below the cutoff frequency pass unchanged
- frequencies above the cutoff frequency are completely removed

However, ideal filters cannot be realized perfectly in practice. Real filters
always have a transition region between passband and stopband. The width of
this transition region depends on the filter design and the filter order.
""")


# An ideal low-pass filter has an abrupt transition from passband to stopband.
# A real low-pass filter has a transition region.
freq = np.linspace(0, 30, 1000)

cutoff = 10
transition_width = 5

ideal_lp = np.where(freq <= cutoff, 1, 0)

real_lp = np.ones_like(freq)
real_lp[freq > cutoff] = np.maximum(
    0,
    1 - (freq[freq > cutoff] - cutoff) / transition_width
)
real_lp[freq > cutoff + transition_width] = 0

fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(freq, ideal_lp, label="Ideal low-pass filter")
ax.plot(freq, real_lp, label="Realistic low-pass filter")

ax.axvline(cutoff, linestyle="--", label="Cutoff frequency")
ax.axvspan(cutoff, cutoff + transition_width, alpha=0.2, label="Transition region")

ax.set_title("Ideal versus realistic low-pass filter")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Magnitude response")
ax.set_ylim(-0.1, 1.1)
ax.grid(True)
ax.legend()

st.pyplot(fig)


st.subheader("Phase response and delay")

st.markdown("""
The phase response describes how the filter shifts signal components in time.

Different phase behaviors are important:

### Zero-phase filtering

A zero-phase filter does not shift the signal in time. The output remains
aligned with the input. However, true zero-phase filtering is acausal, because
it requires information from the past and the future. Therefore, it is only
possible offline.

### Linear-phase filtering

A linear-phase filter shifts all frequency components by the same delay. This
means that the signal is delayed, but its shape is not distorted.

### Nonlinear-phase filtering

A nonlinear-phase filter shifts different frequencies by different amounts.
This can distort the shape of the signal.
""")

st.subheader("Causal and acausal filtering")

st.markdown("""
A **causal filter** only uses the current and past values of a signal. This is
necessary for real-time applications.

Example:

$y[n] = x[n] - x[n-1]$

An **acausal filter** also uses future values of the signal.

Example:

$y[n] = x[n+1] - x[n]$

Acausal filtering is only possible when the full signal is already available.
This is often the case in offline EEG analysis.
""")

st.subheader("FIR and IIR filters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### FIR filters

    FIR means **Finite Impulse Response**.

    FIR filters have an impulse response with a finite number of non-zero
    samples. They can be implemented as a convolution with a finite filter
    kernel.

    Important properties:

    - no feedback
    - finite impulse response
    - always stable
    - can have linear phase
    - often require a higher order for sharp transitions
    """)

with col2:
    st.markdown("""
    ### IIR filters

    IIR means **Infinite Impulse Response**.

    IIR filters use feedback. This means that previous output values are used
    to calculate the current output value.

    Important properties:

    - use feedback
    - theoretically infinite impulse response
    - can achieve sharp filtering with low order
    - often have nonlinear phase
    - stability must be considered
    """)

st.subheader("Difference equation")

st.markdown("""
A digital filter is often described by a difference equation:
""")
st.latex(r"""
y[n] =
\sum_{k=0}^{M} b_k x[n-k]
-
\sum_{k=1}^{N} a_k y[n-k]
""")

st.markdown("""
The $b_k$ coefficients describe the feedforward part of the filter.

The $a_k$ coefficients describe the feedback part of the filter.

For an FIR filter, there is no feedback part (second sum). For an IIR filter, previous
output values are used, which creates feedback.
""")

st.subheader("Transfer function and z-transform")

st.markdown("""
The z-transform is used to describe digital filters mathematically. It allows
us to derive the **transfer function**:
""")

st.latex(r"""
H(z) = \frac{Y(z)}{X(z)}
""")

st.markdown("""
The transfer function describes the relationship between the input and output
of the filter in the z-domain.

It is especially useful for analyzing:

- poles and zeros
- stability
- filter type
- frequency response
""")

st.subheader("Poles, zeros, and stability")

st.markdown("""
The transfer function can be analyzed using a pole-zero plot.

- **Zeros** are frequencies or regions where the filter response is reduced.
- **Poles** are related to amplification and stability.

For causal IIR filters, stability requires all poles to lie inside the unit
circle in the z-plane:

$|z| < 1$

If poles lie on or outside the unit circle, the filter can become unstable.
""")

st.subheader("Filter order")

st.markdown("""
The filter order describes the complexity of the filter.

For FIR filters, the term **taps** refers to the individual filter coefficients.
These coefficients are the values of the FIR filter's impulse response. Each tap
defines how strongly one input sample contributes to the filtered output.

For example, a simple 3-point moving-average filter has three taps:
""")

st.latex(r"""
h[n] = \frac{1}{3}[1, 1, 1]
""")

st.markdown("""
This means that the filter uses three neighboring input samples to calculate
one output sample.

For FIR filters, the filter order is the number of taps minus one:
""")

st.latex(r"""
\text{FIR filter order} = N_{\text{taps}} - 1
""")

st.markdown("""
For IIR filters, the order is related to the number of poles.

A higher filter order usually leads to:

- a sharper transition between passband and stopband
- stronger stopband attenuation
- more delay or stronger transient effects
- higher computational cost
""")

st.subheader("Filter transient")

st.markdown("""
At the beginning of a filtered signal, the output can be distorted because the
filter does not yet have enough previous samples available. This initial
distortion is called a **filter transient**.

Filter transients become more important for higher-order filters.

This is especially relevant when filtering short signal segments. Filtering
only a short segment can create stronger artifacts than filtering the full
signal first and segmenting afterwards.
""")

st.subheader("Relevance for this exercise")

st.markdown("""
In this exercise, FIR and IIR low-pass filters are designed and applied to an
averaged EEG signal.

The analysis will compare:

- FIR and IIR low-pass filters
- different filter orders
- magnitude responses
- phase responses
- transfer function coefficients and second-order sections
- Kaiser-window FIR designs with different transition widths
- filtering the full signal versus filtering only a segment

The goal is to understand how filter design choices influence both the
frequency response and the resulting filtered EEG waveform.
""")