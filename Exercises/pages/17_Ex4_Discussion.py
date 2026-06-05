import streamlit as st
st.subheader("FIR versus IIR filters")

st.markdown("""
The results showed that FIR and IIR filters behave differently in both
magnitude and phase response.

For FIR filters, increasing the number of taps made the transition from
passband to stopband sharper. However, this required a much higher filter order.
The FIR filters showed an approximately linear phase response in the passband,
which means that the waveform shape is mostly preserved, although the signal is
delayed.

For IIR Butterworth filters, a clear low-pass behavior was achieved with much
lower filter orders. Increasing the IIR order made the transition steeper.
However, the phase response was nonlinear, especially around the cutoff
frequency. This means that different frequency components may be shifted by
different amounts.
""")

st.subheader("Advantages and disadvantages")

st.markdown("""
| Filter type | Advantages | Disadvantages |
|---|---|---|
| FIR filter | Stable, approximately linear phase, preserves waveform shape better | Requires more taps, higher order, more delay |
| IIR Butterworth filter | Sharp filtering with low order, computationally efficient, smooth magnitude response | Nonlinear phase, possible waveform distortion, stability must be considered |
""")

st.subheader("Interpretation for EEG filtering")

st.markdown("""
For EEG data, the choice of filter depends on the goal of the analysis.

If preserving the timing and shape of the EEG waveform is important, an FIR
filter is often preferable because of its approximately linear phase behavior.

If the main goal is efficient attenuation of high-frequency components, an IIR
Butterworth filter can be useful because it achieves a strong low-pass effect
with a much lower order.
""")


import streamlit as st

st.header("1. Frequency Response Differences")
st.markdown("""
**How do FIR and IIR filters differ regarding required order and magnitude response?**
* **IIR Filters:** Because they utilize a recursive feedback loop, IIR filters can achieve a highly steep transition band (a sharp cutoff) with a very low filter order. 
* **FIR Filters:** To achieve that same steep transition, an FIR filter requires a massive number of taps (coefficients), resulting in a much higher order and increased computational cost. 

**What differences do you observe in the phase response?**
* **FIR filters:** Introduce a noticeable, consistent delay but perfectly preserve the structural shape of the EEG waveform. This visualizes their **linear phase** property, where all frequencies are delayed by the exact same amount of time.
* **IIR Filters:** Naturally have a **non-linear phase**, particularly near the cutoff frequency. This means different frequencies are delayed by different amounts, which can subtly warp and distort the shape of the EEG signal.
""")

st.header("2. Coefficients vs. Second-Order Sections (SOS)")
st.markdown("""
**Why does $h[n] = b[n]$ for FIR filters?**
The difference equation for an FIR filter has no recursive feedback loop ($y[n-1]$ terms), meaning the denominator coefficient is simply $a = [1]$. If you input a unit impulse ($\delta[n]$ with a single `1` followed by `0`s), the zeros clear out all terms except the feedforward coefficients ($b$). Therefore, the physical array of coefficients directly constitutes the finite impulse response.

**What is the SOS-matrix, and why is it preferred over transfer function coefficients?**
The Second-Order Sections (SOS) matrix breaks down a high-order IIR filter into a cascaded series of smaller, 2nd-order filters (biquads). 
When using standard transfer coefficients (`b`, `a`), the polynomials for high-order filters become extremely sensitive. Floating-point rounding errors in standard 64-bit systems can shift poles outside the unit circle, turning a mathematically stable filter into an unstable one that destroys the signal. The SOS matrix perfectly isolates these roots, guaranteeing numerical stability.
""")

st.header("3. Kaiser Estimation & Transition Widths")
st.markdown("""
**How does the transition width affect filter order and magnitude response?** There is a strict, inversely proportional mathematical trade-off. To achieve a very sharp magnitude response (a steep drop-off between the passband and stopband), the transition width must decrease. However, shrinking this width forces the Kaiser estimation formula to drastically increase the filter order (the number of taps).

**What happens to the filtered signals for very small transition widths?** When the transition width is forced down to extreme limits (e.g., $0.2\text{ Hz}$), the required filter order explodes into the thousands. Applying such a massive filter introduces severe computational lag, extreme phase delay, and violently extends the duration of the filter transient (ringing artifacts) whenever sudden signal changes occur.
""")

st.header("4. Filter Transients & Segmentation")
st.markdown("""
**What do you observe in the filtered signals when filtering the full versus the segmented signal?** When the isolated $2\text{-second}$ segment is filtered independently, a massive oscillation (ringing) completely distorts the first $0.5\text{ seconds}$ of the data. When the full $10\text{-second}$ signal is filtered first and then segmented, this distortion is completely absent, and the filter perfectly tracks the raw signal through the exact same time window.

**What is the smarter approach: filtering first then segmenting, or the other way around? Why?** **Filtering first, then segmenting** is the only mathematically sound approach. 
Digital filters utilize a memory buffer of past samples to compute the current output. If a signal is segmented prior to filtering, the algorithm encounters a hard boundary at $t=0$ and assumes the historical state of the signal was exactly `0.0`. When the first physical EEG voltage arrives, it acts as a massive step-input shock to the empty memory buffer. The filter wildly overcompensates, resulting in the violent "ringing" transient. 
By filtering the full continuous signal first, the initial shock occurs far outside the region of interest. By the time the filter's sliding window reaches the critical segment, its memory buffer is populated with accurate historical state data, ensuring perfect stability.
""")

st.caption("DSP Exercise 4 * FH Joanneum * 2026")