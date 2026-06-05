import scipy.signal as signal
import numpy as np

def apply_lfilter(b, a, data):
    """Applies standard transfer function filtering."""
    return signal.lfilter(b, a, data)

def apply_sosfilt(sos, data):
    """Applies second-order sections filtering."""
    return signal.sosfilt(sos, data)

def design_kaiser(cutoff: float, transition_width: float, fs: int, ripple_db: float = 60.0):
    """
    Estimates order and designs an FIR filter using the Kaiser window method.
    """
    nyq = 0.5 * fs #Ntyquist frequency - limit
    width_norm = transition_width / nyq
    
    #number of taps estimation and the Kaiser beta parameter
    num_taps, beta = signal.kaiserord(ripple_db, width_norm)
    
    #we ensure taps are odd to prevent a zero at the Nyquist frequency (Type I FIR)
    if num_taps % 2 == 0:
        num_taps += 1
        
    normalized_cutoff = cutoff / nyq
    b = signal.firwin(num_taps, normalized_cutoff, window=('kaiser', beta))
    
    return b, num_taps