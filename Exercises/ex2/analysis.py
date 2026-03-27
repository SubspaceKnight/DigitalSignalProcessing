import numpy as np
import pandas as pd

from Exercises.utils.shared import compute_fs_stats


#DFT from the definition
def dft(x: np.ndarray) -> np.ndarray:
    #X[k] = sum_{n=0}^{N-1}  x[n] * exp(-j * 2*pi * k * n / N)
    x = np.asarray(x, dtype=float)
    N = len(x)
    n = np.arange(N)
    k = n[:, None]                           
    W = np.exp(-2j * np.pi * k * n / N)       
    return W @ x                              


def dft_windowed(x: np.ndarray, max_n: int = 4096) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N <= max_n:
        return dft(x)
    step  = N // max_n
    x_sub = x[::step]
    return dft(x_sub)


#Frequency axis + spectral helpers
def frequency_bins(N: int, fs: float) -> np.ndarray:
    #f[k] = k * fs / N, for k = 0, 1, ..., N//2
    return np.arange(N // 2 + 1) * fs / N


def magnitude_spectrum(X: np.ndarray, normalise: bool = True) -> np.ndarray:
    """
    With normalise=True:
    |X[k]| / N  for k = 0 and k = N//2  (DC, Nyquist)
    2*|X[k]|/N  for 0 < k < N//2          (fold negative side in)
    """
    N   = len(X)
    Xh  = X[: N // 2 + 1]
    mag = np.abs(Xh)
    if normalise:
        mag = mag / N
        mag[1 : N // 2] *= 2
    return mag


def power_spectrum(X: np.ndarray, normalise: bool = True) -> np.ndarray:
    return magnitude_spectrum(X, normalise=normalise) ** 2


#(task 1) Sampling-rate + Nyquist statistics  
def sampling_stats(sig_df: pd.DataFrame) -> dict:
    """
    All sampling-rate statistics derived from the 'time_s' column.
    Delegates arithmetic to shared.compute_fs_stats.

    Extra aliases (fs_median_hz, fs_std_hz, …) are added so the Streamlit
    pages can use either naming style.
    """
    stats = compute_fs_stats(sig_df["time_s"])
    stats["fs_mean_hz"]   = stats["mean_hz"]
    stats["fs_median_hz"] = stats["median_hz"]
    stats["fs_std_hz"]    = stats["std_hz"]
    return stats


#(task 2) Full-signal DFT pipeline  
def run_full_spectrum(
    sig_df: pd.DataFrame,
    fs: float,
    max_n: int = 4096) -> dict:

    x     = sig_df["amplitude"].values
    N_raw = len(x)
    fs_used = fs

    if N_raw > max_n:
        step    = N_raw // max_n
        x       = x[::step]
        fs_used = fs / step

    X     = dft(x)
    N     = len(X)
    freqs = frequency_bins(N, fs_used)
    mag   = magnitude_spectrum(X)
    pwr   = power_spectrum(X)

    return {
        "X":         X,
        "freqs":     freqs,
        "magnitude": mag,
        "power":     pwr,
        "N_used":    N,
        "N_raw":     N_raw,
        "fs_used":   round(fs_used, 4),
    }
def run_segment_spectrum(
    sig_df: pd.DataFrame,
    fs: float,
    n_samples: int = 4096,
) -> dict:
    """
    Compute the DFT on the FIRST n_samples raw samples at the TRUE fs.
 
    This is the correct way to get a spectrum that shows 0 … fs/2 Hz.
    Sub-sampling the full signal (as done in run_full_spectrum) reduces the
    effective fs and shrinks the visible frequency range — which is misleading.
 
    Trade-off: frequency resolution = fs / n_samples.
      n_samples=4096, fs=256 Hz → Δf = 0.0625 Hz (fine enough to resolve 50 Hz)
 
    Returns
    -------
    dict: X, freqs, magnitude, power, N_used, fs
    """
    x = sig_df["amplitude"].values[:n_samples]
    X = dft(x)
    N = len(X)
    freqs = frequency_bins(N, fs)
    mag   = magnitude_spectrum(X)
    pwr   = power_spectrum(X)
 
    return {
        "X":         X,
        "freqs":     freqs,
        "magnitude": mag,
        "power":     pwr,
        "N_used":    N,
        "fs":        fs,
    }


#(task 2) Spectral peak detection + noise/signal classification  
def find_spectral_peaks(
    freqs: np.ndarray,
    mag: np.ndarray,
    n_peaks: int       = 10,
    min_freq_hz: float = 0.5) -> pd.DataFrame:
    mask = freqs >= min_freq_hz
    f_m  = freqs[mask]
    m_m  = mag[mask]

    idx  = np.argsort(m_m)[::-1][:n_peaks]
    rows = [
        {"Frequency (Hz)": round(float(f_m[i]), 3),
         "Magnitude":      round(float(m_m[i]), 6)}
        for i in idx
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("Frequency (Hz)")
        .reset_index(drop=True)
    )


def classify_peaks(peaks_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in peaks_df.iterrows():
        f = r["Frequency (Hz)"]
        m = r["Magnitude"]

        if f < 0.5:
            clf    = "DC / drift"
            source = "Electrode offset or very slow baseline drift"
        elif abs(f - 50) <= 2 or abs(f - 60) <= 2:
            clf    = "Noise - mains interference"
            source = "50/60 Hz power-line pickup"
        elif any(abs(f - n * 50) <= 2 for n in range(2, 10)) or \
             any(abs(f - n * 60) <= 2 for n in range(2, 10)):
            clf    = "Noise - mains harmonic"
            source = "Harmonic of 50/60 Hz power-line noise"
        else:
            clf    = "Candidate signal"
            source = "Physiological oscillation or stimulus-related activity"

        rows.append({
            "Frequency (Hz)": f,
            "Magnitude":      round(m, 6),
            "Classification": clf,
            "Likely source":  source,
        })

    return pd.DataFrame(rows)


#(task 4) Downsampling analysis  
def downsample_spectrum(
    sig_df: pd.DataFrame,
    factor: int,
    fs_orig: float,
    n_samples: int = 4096, ) -> dict:
    """
    Downsample the signal by *factor* (no anti-aliasing filter), then compute
    the DFT on the first n_samples of the downsampled signal.
 
    The new sampling rate is fs_new = fs_orig / factor.
    The new Nyquist is fs_new / 2.
 
    Any component that was above the new Nyquist in the original signal will
    alias — it folds into the spectrum at a wrong lower frequency.
 
    Returns
    -------
    dict: freqs, magnitude, fs_new, nyquist_new, n_used, factor
    """
    x_ds   = sig_df["amplitude"].values[::factor]   # keep every factor-th sample
    fs_new = fs_orig / factor
 
    # Take the first n_samples of the downsampled signal
    x_seg = x_ds[:n_samples]
    X     = dft(x_seg)
    N     = len(X)
    freqs = frequency_bins(N, fs_new)
    mag   = magnitude_spectrum(X)
 
    return {
        "freqs":       freqs,
        "magnitude":   mag,
        "fs_new":      round(fs_new, 4),
        "nyquist_new": round(fs_new / 2, 4),
        "n_used":      N,
        "factor":      factor,
    }
 
 
def aliased_frequency(f_true: float, fs_new: float) -> float:
    """
    Predict where a component at f_true Hz will appear after downsampling
    to fs_new Hz (without an anti-aliasing filter).
 
    The folding formula:
        f_alias = | ((f_true + fs_new/2) mod fs_new) - fs_new/2 |
 
    This wraps f_true into the range [0, fs_new/2] accounting for the
    periodic nature of the DFT.
    """
    nyq = fs_new / 2
    return round(abs(((f_true + nyq) % fs_new) - nyq), 4)
