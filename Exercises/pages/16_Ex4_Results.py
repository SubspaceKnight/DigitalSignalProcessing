import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

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

st.caption("DSP Exercise 4 * FH Joanneum * 2026")