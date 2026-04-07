import numpy as np
import streamlit as st
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR.parent / "data" / "audio" 

@st.cache_data
def load_audio(filename: str, mono: bool = True) -> dict:
    """
    Supports: .wav, .mp3, .flac, .ogg.
    librosa is used ONLY for I/O and resampling - NOT for analysis.

    Params:
    filename - filename relative to DATA_DIR ("song_a.wav").
    mono - if true convert to mono.

    Returns:
    dict
        Keys: samples (np.ndarray, float32), sr (abbr "SampleRate", int, Hz), duration (float, s),
        n_samples (int), filename (str), channels (int, pre-mono count).
    """
    try:
        import librosa
    except ImportError:
        st.error("librosa is not installed. Run: uv add librosa")
        return {}

    path = DATA_DIR / filename
    if not path.exists():
        st.error(f"Audio file not found: {path}")
        return {}

    samples, sr = librosa.load(str(path), sr=None, mono=mono)

    #librosa returns (channels, samples) for stereo when mono=False, or (samples,) for mono. 
    #normalise to always be (samples,) or (2, samples)?
    if samples.ndim == 1:
        channels = 1
    else:
        channels = samples.shape[0]

    if mono and samples.ndim > 1:
        samples = samples.mean(axis=0)

    samples = samples.astype(np.float32)

    return {
        "samples":   samples,
        "sr":        int(sr),
        "duration":  float(len(samples) / sr),
        "n_samples": len(samples),
        "filename":  filename,
        "channels":  channels,
    }


@st.cache_data
def list_audio_files() -> list[str]:
    """Return all audio files present in DATA_DIR (sorted)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"}
    return sorted(
        p.name for p in DATA_DIR.iterdir()
        if p.suffix.lower() in extensions
    )


def audio_info_df(audio: dict) -> dict:
    """Return a flat dict of human-readable metadata for display in st.metric / st.dataframe."""
    if not audio:
        return {}
    return {
        "File":        audio["filename"],
        "Sample_rate": f"{audio['sr']} Hz",
        "Duration":    f"{audio['duration']:.2f} s",
        "Samples":     f"{audio['n_samples']:,}",
        "Channels":    audio["channels"],
        "Nyquist":     f"{audio['sr'] // 2} Hz", #can be useful for filter design or be removed
    }