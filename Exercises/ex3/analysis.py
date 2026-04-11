import numpy as np
import pandas as pd


#Window functions!

WINDOW_FUNCTIONS = ["rectangular", "hann", "hamming", "blackman", ] 

def get_window(name: str, length: int) -> np.ndarray:
    """
    Returns a window of the given length.

    Params:
        name:  'rectangular', 'hann', 'hamming', 'blackman',
        length: number of samples

    Returns:
    np.ndarray of shape (length,), values in [0, 1]
    """
    n = np.arange(length)
    if name == "hann":
        return 0.5 * (1 - np.cos(2 * np.pi * n / (length - 1)))
    elif name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (length - 1))
    elif name == "blackman":
        return (
            0.42
            - 0.50 * np.cos(2 * np.pi * n / (length - 1))
            + 0.08 * np.cos(4 * np.pi * n / (length - 1))
        )
    elif name == "rectangular": #easiest to implement but not recommended for real analysis due to spectral leakage
        return np.ones(length)
    else:
        raise ValueError(f"Unknown window: {name}. Choose from {WINDOW_FUNCTIONS}")


#DFT (reused from Ex2 - same definition-based implementation)
# def dft(x: np.ndarray) -> np.ndarray:
#     x = np.asarray(x, dtype=float)
#     N = len(x)
#     n = np.arange(N)
#     k = n[:, None]
#     W = np.exp(-2j * np.pi * k * n / N)
#     return W @ x
def dft(x, W=None):
    x = np.asarray(x, dtype=complex)
    N = len(x)

    if W is None:
        n = np.arange(N)
        k = n.reshape((N, 1))
        W = np.exp(-2j * np.pi * k * n / N)

    return W @ x


# STFT - implemented from scratch
def stft(
    samples:     np.ndarray, 
    sr:          int, #sample rate in Hz of course, needed for the frequency axis and to convert hop_length from samples to seconds
    window_name: str  = "hann", #default to the most common window type for general analysis, but feel free to experiment with others
    win_length:  int  = 1024, #typical values are 512, 1024, 2048 - longer windows give better frequency resolution but worse time resolution, and vice versa
    hop_length:  int  = 512, #typical values are 256, 512, 1024 - usually set to win_length // 2 for 50% overlap, but can be adjusted for more/less overlap
) -> dict:
    """
    Short-Time Fourier Transform.
    The signal is split into partially overlapping frames. Each frame is multiplied by
    the chosen window function and then transformed with our own DFT or can be used normal.

    Params
        samples: mono audio array (float32)
        sr: sample rate in Hz
        window_name: window function name (see WINDOW_FUNCTIONS)
        win_length: frame length in samples
        hop_length: hop size in samples (= win_length - overlap)

    Returns
    dict with keys:
        S - complex STFT matrix, shape (n_freqs, n_frames)
        n_freqs = win_length // 2 + 1  (one-sided)
        magnitude - |S|, shape (n_freqs, n_frames)
        power - |S|**2,shape (n_freqs, n_frames)
        freqs - frequency axis in Hz, shape (n_freqs,)
        times - frame centre times in seconds, shape (n_frames,)
        win_length - int
        hop_length - int
        window_name - str
        sr - int
        n_frames - int
        n_freqs - int
    """


    samples = np.asarray(samples, dtype=float)
    window = get_window(window_name, win_length)
    n_freqs = win_length // 2 + 1

    #frame the signal
    #Pad so the last frame is complete
    n_pad = win_length  #pad start so frame 0 is centred at t=0
    samples_p = np.pad(samples, (n_pad, win_length), mode="constant")

    #Number of complete frames
    n_frames = 1 + (len(samples_p) - win_length) // hop_length

    #allocate output array
    S = np.zeros((n_freqs, n_frames), dtype=complex)

    n = np.arange(win_length)
    k = n.reshape((win_length, 1))
    W_dft = np.exp(-2j * np.pi * k * n / win_length)

    #main loop I guess, could be optimised with strided windows and matrix multiplication but this is clearer for learning purposes
    for m in range(n_frames):
        start = m * hop_length
        frame = samples_p[start : start + win_length]
        frame_w = frame * window # apply window

        X = dft(frame_w, W=W_dft)
        S[:, m] = X[:n_freqs]

    #axes
    freqs = np.arange(n_freqs) * sr / win_length
    #frame centre times (accounting for the padding)
    times = (np.arange(n_frames) * hop_length - n_pad + win_length / 2) / sr

    mag = np.abs(S)
    pwr = mag ** 2

    return {
        "S":           S,
        "magnitude":   mag,
        "power":       pwr,
        "freqs":       freqs,
        "times":       times,
        "win_length":  win_length,
        "hop_length":  hop_length,
        "window_name": window_name,
        "sr":          sr,
        "n_frames":    n_frames,
        "n_freqs":     n_freqs,
    }


# Spectral representations
def to_db(magnitude: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    """
    Convert magnitude to decibel scale.
    dB[k, m] = 20 * log10(max(|S[k,m]|, amin) / ref)
    """
    return 20.0 * np.log10(np.maximum(magnitude, amin) / ref)


def to_mel(
    power:    np.ndarray,
    sr:       int,
    freqs:    np.ndarray,
    n_mels:   int = 128,
    fmin:     float = 0.0,
    fmax:     float = None,
) -> dict:
    """
    Project power spectrum onto the Mel scale.
    The Mel filterbank is built here from scratch using the standard HTK formula. No librosa mel functions are used.

    Params
        power: (n_freqs, n_frames) power spectrogram
        sr: sample rate
        freqs: frequency axis in Hz (n_freqs,)
        n_mels: number of Mel filters
        fmin: lowest Mel filter centre (Hz)
        fmax: highest Mel filter centre (Hz); defaults to sr/2

    Returns
    dict:
        mel_spec - (n_mels, n_frames) Mel power spectrogram
        mel_freqs - (n_mels,) centre frequencies of Mel bins in Hz
        filterbank - (n_mels, n_freqs) the filterbank matrix
    """
    
    if fmax is None:
        fmax = float(sr) / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    #Build triangular filters
    n_freqs = len(freqs)
    fb = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        f_lo = hz_points[m - 1]
        f_cent = hz_points[m]
        f_hi = hz_points[m + 1]
        for k, f in enumerate(freqs):
            if f_lo <= f <= f_cent:
                fb[m - 1, k] = (f - f_lo) / (f_cent - f_lo + 1e-12)
            elif f_cent < f <= f_hi:
                fb[m - 1, k] = (f_hi - f) / (f_hi - f_cent + 1e-12)

    mel_spec = fb @ power # (n_mels, n_frames)
    mel_freqs = hz_points[1:-1]

    return {
        "mel_spec":   mel_spec,
        "mel_freqs":  mel_freqs,
        "filterbank": fb,
    }


#(Task 3)Band energy features  
def compute_band_energy(
    stft_result: dict,
    bands:       list[tuple[float, float]], #list of (f_low, f_high) tuples in Hz e.g. [(0, 500), (500, 4000), (4000, 8000), (8000, X)]
) -> dict:
    """
    Compute per-band energy features as defined in the assignment sheet.
    For each band B = {k : f_low ≤ freqs[k] < f_high}:
        EB[m] = Σ_{k∈B} |X[k,m]|**2 (band energy per frame)
        E^_B[m] = EB[m] / Σ_k |X[k,m]|**2 (relative band energy per frame)
        <E^_B> = (1/M) Σ_m E^_B[m] (time-averaged relative band energy)

    Params
        stft_result: output of stft()
        bands: list of (f_low, f_high) tuples in Hz

    Returns
    dict: one entry per band label, each containing:
        EB - (n_frames,) band energy per frame
        EB_rel - (n_frames,) relative band energy per frame
        EB_mean - float, time-averaged relative band energy
        label - str, e.g. "0-500 Hz"
        indices - list of frequency bin indices in this band
    """
    freqs = stft_result["freqs"]
    power = stft_result["power"] # (n_freqs, n_frames)
    total = power.sum(axis=0) + 1e-12 # (n_frames,) total power per frame

    results = {}
    for f_low, f_high in bands:
        idx   = np.where((freqs >= f_low) & (freqs < f_high))[0]
        label = f"{int(f_low)}-{int(f_high)} Hz"
        if len(idx) == 0:
            results[label] = {
                "EB": np.zeros(power.shape[1]),
                "EB_rel": np.zeros(power.shape[1]),
                "EB_mean": 0.0,
                "label": label,
                "indices": [],
            }
            continue
        EB = power[idx, :].sum(axis=0) # (n_frames,)
        EB_rel = EB / total # (n_frames,)
        EB_mean = float(EB_rel.mean())

        results[label] = {
            "EB":      EB,
            "EB_rel":  EB_rel,
            "EB_mean": EB_mean,
            "label":   label,
            "indices": idx.tolist(),
        }
    return results


#Additional metric - Spectral Centroid  (one of several options to pick from)
def spectral_centroid(stft_result: dict) -> np.ndarray:
    """
    Spectral centroid per frame:
        SC[m] = Σ_k (f_k · |X[k,m]|) / Σ_k |X[k,m]|

    Describes the 'centre of mass' of the spectrum — a high value means the energy is concentrated 
    at high frequencies (bright/noisy timbre), a low value means predominantly low-frequency content (bass-heavy).

    Returns
    -------
    np.ndarray of shape (n_frames,) in Hz
    """
    freqs = stft_result["freqs"] # (n_freqs,)
    mag = stft_result["magnitude"] # (n_freqs, n_frames)

    num = (freqs[:, None] * mag).sum(axis=0) # (n_frames,)
    den = mag.sum(axis=0) + 1e-12
    return num / den


def spectral_flatness(stft_result: dict) -> np.ndarray:
    """
    Spectral flatness (Wiener entropy) per frame:
        SF[m] = geometric_mean(|X[k,m]|**2) / arithmetic_mean(|X[k,m]|**2)

    Values near 1 -> noise-like (flat spectrum).
    Values near 0 -> tonal (energy concentrated in a few bins).

    Returns
    np.ndarray of shape (n_frames,)
    """
    power = stft_result["power"] # (n_freqs, n_frames)
    power = np.maximum(power, 1e-12)
    log_mean = np.log(power).mean(axis=0)
    arith_mean = power.mean(axis=0) + 1e-12
    return np.exp(log_mean) / arith_mean


# Utility functions for displaying results in Streamlit
#(not the main focus of the assignment)
def band_energy_summary_df(band_results: dict) -> pd.DataFrame:
    """Compact summary table of time-averaged relative band energies."""
    rows = [
        {
            "Band":                     v["label"],
            "Time-avg relative energy": round(v["EB_mean"], 4),
            "Frequency bins":            len(v["indices"]),
        }
        for v in band_results.values()
    ]
    return pd.DataFrame(rows)