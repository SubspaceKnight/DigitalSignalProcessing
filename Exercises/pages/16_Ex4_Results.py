import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from ex4.helper import load_eeg_excel
from ex4.analysis import apply_lfilter, apply_sosfilt, design_kaiser
# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
# Wide layout gives more space for plots and interactive controls.
st.set_page_config(
    page_title="Results",
    layout="wide"
)

st.title("Results - FIR and IIR Filter Responses")

# ------------------------------------------------------------
# General parameters
# ------------------------------------------------------------
# Sampling frequency from the assignment.
# The EEG signal was sampled at 512 Hz.
fs = 512

# Cutoff frequency required by the assignment.
# All filters are designed as low-pass filters with fc = 10 Hz.
cutoff = 10

# Number of frequency points used by scipy.signal.freqz.
# This only controls the smoothness/resolution of the plotted frequency response.
worN = 4096

# Small numerical value to avoid log10(0) when converting magnitude to dB.
eps = 1e-12

# ------------------------------------------------------------
# Sidebar settings
# ------------------------------------------------------------
st.sidebar.header("Filter settings")

# FIR tap numbers to compare.
# These values cover low, medium, and high FIR filter orders.
# Odd tap numbers are commonly used for FIR low-pass filters because they allow
# a symmetric impulse response with a clear center sample.
fir_taps = st.sidebar.multiselect(
    "FIR number of taps",
    options=[21, 51, 101, 201, 401],
    default=[51, 101, 201],
    help="For FIR filters, order = number of taps - 1."
)

# IIR filter orders to compare.
# Lower orders are computationally simple, higher orders create steeper
# transitions but stronger phase effects.
iir_orders = st.sidebar.multiselect(
    "IIR Butterworth order",
    options=[1, 2, 4, 6, 8, 10],
    default=[2, 4, 8],
    help="For IIR Butterworth filters, the order is set directly."
)

st.sidebar.header("Direct comparison")

# Single FIR filter for direct comparison.
selected_fir_taps = st.sidebar.selectbox(
    "Selected FIR taps",
    options=[21, 51, 101, 201, 401],
    index=2
)

# Single IIR filter for direct comparison.
selected_iir_order = st.sidebar.selectbox(
    "Selected IIR order",
    options=[1, 2, 4, 6, 8, 10],
    index=2
)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def magnitude_db(h):
    """Convert complex frequency response to magnitude in dB."""
    return 20 * np.log10(np.maximum(np.abs(h), eps))


def phase_rad(h):
    """Calculate unwrapped phase response in radians."""
    return np.unwrap(np.angle(h))


def design_fir(numtaps):
    """
    Design an FIR low-pass filter.
    scipy.signal.firwin 
    numtaps defines the number of FIR coefficients.
    cutoff is given in Hz because fs is also provided.
    """
    return signal.firwin(
        numtaps=numtaps,
        cutoff=cutoff,
        fs=fs,
        pass_zero="lowpass"
    )


def design_iir(order):
    """
    Design an IIR Butterworth low-pass filter.
    scipy.signal.butter
    N is the IIR filter order.
    output='ba' returns transfer function coefficients b and a.
    """
    return signal.butter(
        N=order,
        Wn=cutoff,
        btype="lowpass",
        fs=fs,
        output="ba"
    )


def get_response(b, a):
    """
    Calculate frequency response.

    fs=fs makes scipy return the frequency axis in Hz.
    """
    return signal.freqz(
        b=b,
        a=a,
        worN=worN,
        fs=fs
    )


# ------------------------------------------------------------
# Short task information
# ------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Sampling rate", f"{fs} Hz")
col2.metric("Cutoff frequency", f"{cutoff} Hz")
col3.metric("Selected FIR order", selected_fir_taps - 1)
col4.metric("Selected IIR order", selected_iir_order)

st.caption(
    "FIR filters were designed with scipy.signal.firwin. "
    "IIR Butterworth filters were designed with scipy.signal.butter."
)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "FIR filters",
    "IIR Butterworth filters",
    "Direct comparison"
])

# ============================================================
# TAB 1: FIR filters
# ============================================================
with tab1:
    st.subheader("FIR low-pass filters")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # FIR magnitude response
    # --------------------------------------------------------
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))

        for taps in fir_taps:
            b_fir = design_fir(taps)
            w, h = get_response(b_fir, [1.0])

            ax.plot(
                w,
                magnitude_db(h),
                label=f"{taps} taps, order {taps - 1}"
            )

        ax.axvline(cutoff, linestyle="--", label="Cutoff frequency")
        ax.set_title("FIR magnitude response")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_xlim(0, 60)
        ax.set_ylim(-100, 5)
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

        st.caption(
            "More taps produce a narrower transition region and stronger attenuation above the cutoff."
        )

    # --------------------------------------------------------
    # FIR phase response
    # --------------------------------------------------------
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))

        for taps in fir_taps:
            b_fir = design_fir(taps)
            w, h = get_response(b_fir, [1.0])

            ax.plot(
                w,
                phase_rad(h),
                label=f"{taps} taps, order {taps - 1}"
            )

        ax.axvline(cutoff, linestyle="--", label="Cutoff frequency")
        ax.set_title("FIR phase response")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Phase [rad]")
        ax.set_xlim(0, 60)
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

        st.caption(
            "The FIR phase is approximately linear in the passband; higher tap numbers increase delay."
        )

    st.divider()

    # --------------------------------------------------------
    # FIR selected filter details
    # --------------------------------------------------------
    selected_b_fir = design_fir(selected_fir_taps)

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected FIR taps", selected_fir_taps)
    c2.metric("Selected FIR order", selected_fir_taps - 1)
    c3.metric("Number of coefficients", len(selected_b_fir))

# ============================================================
# TAB 2: IIR filters
# ============================================================
with tab2:
    st.subheader("IIR Butterworth low-pass filters")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # IIR magnitude response
    # --------------------------------------------------------
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))

        for order in iir_orders:
            b_iir, a_iir = design_iir(order)
            w, h = get_response(b_iir, a_iir)

            ax.plot(
                w,
                magnitude_db(h),
                label=f"Order {order}"
            )

        ax.axvline(cutoff, linestyle="--", label="Cutoff frequency")
        ax.set_title("IIR Butterworth magnitude response")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_xlim(0, 60)
        ax.set_ylim(-100, 5)
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

        st.caption(
            "Higher IIR orders make the transition from passband to stopband steeper."
        )

    # --------------------------------------------------------
    # IIR phase response
    # --------------------------------------------------------
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))

        for order in iir_orders:
            b_iir, a_iir = design_iir(order)
            w, h = get_response(b_iir, a_iir)

            ax.plot(
                w,
                phase_rad(h),
                label=f"Order {order}"
            )

        ax.axvline(cutoff, linestyle="--", label="Cutoff frequency")
        ax.set_title("IIR Butterworth phase response")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Phase [rad]")
        ax.set_xlim(0, 60)
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

        st.caption(
            "The IIR phase response is nonlinear, especially around the cutoff region."
        )

    st.divider()

    # --------------------------------------------------------
    # IIR selected filter details
    # --------------------------------------------------------
    selected_b_iir, selected_a_iir = design_iir(selected_iir_order)

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected IIR order", selected_iir_order)
    c2.metric("b coefficients", len(selected_b_iir))
    c3.metric("a coefficients", len(selected_a_iir))

# ============================================================
# TAB 3: Direct comparison
# ============================================================
with tab3:
    st.subheader("Direct comparison")

    # Design selected filters.
    b_fir = design_fir(selected_fir_taps)
    w_fir, h_fir = get_response(b_fir, [1.0])

    b_iir, a_iir = design_iir(selected_iir_order)
    w_iir, h_iir = get_response(b_iir, a_iir)

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # Direct magnitude comparison
    # --------------------------------------------------------
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(
            w_fir,
            magnitude_db(h_fir),
            label=f"FIR: {selected_fir_taps} taps, order {selected_fir_taps - 1}"
        )

        ax.plot(
            w_iir,
            magnitude_db(h_iir),
            label=f"IIR Butterworth: order {selected_iir_order}"
        )

        ax.axvline(cutoff, linestyle="--", label="Cutoff frequency")
        ax.set_title("Magnitude response comparison")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_xlim(0, 60)
        ax.set_ylim(-100, 5)
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

    # --------------------------------------------------------
    # Direct phase comparison
    # --------------------------------------------------------
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(
            w_fir,
            phase_rad(h_fir),
            label=f"FIR: {selected_fir_taps} taps, order {selected_fir_taps - 1}"
        )

        ax.plot(
            w_iir,
            phase_rad(h_iir),
            label=f"IIR Butterworth: order {selected_iir_order}"
        )

        ax.axvline(cutoff, linestyle="--", label="Cutoff frequency")
        ax.set_title("Phase response comparison")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Phase [rad]")
        ax.set_xlim(0, 60)
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

    st.caption(
        "The selected IIR filter reaches a clear low-pass response with a lower order, "
        "while the FIR filter shows a more linear phase response in the passband."
    )

st.divider()

from pathlib import Path
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR.parent / "data"
DATA_PATH = DATA_DIR / "data_s05_s1_fcz.xlsx"
eeg_data = load_eeg_excel(DATA_PATH)

if eeg_data is not None:
    time = eeg_data['time']
    signal_raw = eeg_data['signal']
    
    tab4, tab5 = st.tabs(["Signal Filtering & SOS Matrix", "Kaiser Estimation & Transients"])
    
    with tab4:
        st.subheader("Applying Filters to the EEG Signal")
        
        #filters design
        b_fir = design_fir(selected_fir_taps)
        b_iir, a_iir = design_iir(selected_iir_order)
        sos_iir = signal.butter(selected_iir_order, cutoff, btype="lowpass", fs=fs, output="sos")
        
        #applying filters
        filtered_fir = apply_lfilter(b_fir, [1.0], signal_raw) #zero for feedbacks except the first one at 0 sec, only feedforward coefficients
        filtered_iir_ba = apply_lfilter(b_iir, a_iir, signal_raw)
        filtered_iir_sos = apply_sosfilt(sos_iir, signal_raw) #we use SOS (Second-Order Sections) Matrix for the IIR filter to ensure numerical stability, especially for higher orders. The SOS format breaks down the filter into cascaded biquads, which prevents issues with pole-zero sensitivity that can arise with direct transfer function coefficients. This is particularly important for IIR filters, as they can become unstable if the poles are not handled correctly, especially when implemented with finite precision arithmetic (64-bit machines).
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time, signal_raw, alpha=0.3, color='gray', label="Raw EEG")
        ax.plot(time, filtered_fir, label=f"FIR ({selected_fir_taps} taps)")
        ax.plot(time, filtered_iir_ba, linestyle="--", label=f"IIR (lfilter, Order {selected_iir_order})")
        ax.plot(time, filtered_iir_sos, linestyle=":", label=f"IIR (sosfilt, Order {selected_iir_order})")
        
        ax.set_title("Filtered EEG Waveforms")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)
        
        st.info("**Observation:** FIR filters introduce a consistent delay (linear phase shift). High-order IIR filters applied via standard transfer coefficients (`b`, `a`) may exhibit numerical instability, which is why the SOS matrix is preferred.")

    with tab5:
        st.subheader("Kaiser Window Estimation & Transient Effects")
        #The transition width is the gap between "passing" a signal and "stopping" it. A smaller transition width means a sharper cutoff, which requires a higher filter order. When we ask for a very narrow transition (e.g. 0.2Hz), the Kaiser method estimates that we need thousands of taps to achieve that sharpness. Applying such a high-order FIR filter to a signal can cause significant ringing artifacts (Gibbs phenomenon) and an extreme delay, especially if the filter is applied to a short segment without proper state initialization.
        #In simple words - if we ask kaiserord for 0.2 Hz transition width, it will divide the sampling frequency (512 Hz) by 0.2 Hz, which gives us 2560. This means we would need an FIR filter with around 2560 taps to achieve that narrow transition, which is impractical and leads to severe artifacts when applied to real signals.
        #Transition width is the frequency 'gray' range/zone which defines the soft boarders between allowed and banned frequencies. E.g. if we design a 10Hz low-pass filter, we want 9.9Hz to pass perfectly, and 10.1Hz to be completely blocked. 
        widths = [5.0, 2.0, 1.0, 0.2] #Transition widths in Hz for Kaiser filter design. Smaller widths require higher orders.
        selected_width = st.radio("Select Transition Width [Hz]:", widths, horizontal=True)
        
        #design Kaiser filter
        b_kaiser, estimated_taps = design_kaiser(cutoff, selected_width, fs)
        
        st.metric(f"Estimated Order for {selected_width} Hz transition", estimated_taps - 1)
        
        #applying to full signal
        filtered_full = apply_lfilter(b_kaiser, [1.0], signal_raw)
        #extracting last 2 seconds from the already filtered signal
        segment_filtered_first = filtered_full[-int(2 * fs):]
        
        #appplying only to last 2 seconds
        segment_raw = signal_raw[-int(2 * fs):]
        segment_filtered_second = apply_lfilter(b_kaiser, [1.0], segment_raw)
        time_segment = time[-int(2 * fs):]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_segment, segment_raw, alpha=0.3, color='gray', label="Raw Segment")
        ax.plot(time_segment, segment_filtered_first, label="Filtered Full -> Segmented")
        ax.plot(time_segment, segment_filtered_second, linestyle="--", label="Segmented -> Filtered")
        
        ax.set_title(f"Transient Effects (Kaiser FIR, Order {estimated_taps-1})")
        ax.set_xlabel("Time [s]")
        ax.legend()
        st.pyplot(fig)
        
        st.warning("**Transient Risk:** Notice the massive initial distortion when filtering the isolated segment with a high-order filter. The filter lacks previous state history.")


st.caption("DSP Exercise 4 * FH Joanneum * 2026")