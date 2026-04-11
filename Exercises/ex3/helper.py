import numpy as np
import streamlit as st
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "audio"

@st.cache_data
def load_audio(filename: str, mono: bool = True) -> dict:
    try:
        import librosa
    except Exception as e:
        st.error(f"Could not import librosa: {type(e).__name__}: {e}")
        return {}

    path = DATA_DIR / filename
    if not path.exists():
        st.error(f"Audio file not found: {path}")
        return {}

    try:
        samples, sr = librosa.load(str(path), sr=None, mono=mono)
    except Exception as e:
        st.error(f"Could not load audio file '{filename}': {type(e).__name__}: {e}")
        return {}

    if samples.ndim == 1:
        channels = 1
    else:
        channels = samples.shape[0]

    if mono and samples.ndim > 1:
        samples = samples.mean(axis=0)

    samples = samples.astype(np.float32)

    return {
        "samples": samples,
        "sr": int(sr),
        "duration": float(len(samples) / sr),
        "n_samples": len(samples),
        "filename": filename,
        "channels": channels,
    }


@st.cache_data
def list_audio_files() -> list[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"}
    return sorted(
        p.name for p in DATA_DIR.iterdir()
        if p.suffix.lower() in extensions
    )


def audio_info_df(audio: dict) -> dict:
    if not audio:
        return {}
    return {
        "File": audio["filename"],
        "Sample_rate": f"{audio['sr']} Hz",
        "Duration": f"{audio['duration']:.2f} s",
        "Samples": f"{audio['n_samples']:,}",
        "Channels": audio["channels"],
        "Nyquist": f"{audio['sr'] // 2} Hz",
    }