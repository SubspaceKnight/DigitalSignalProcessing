import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ex3 import helper, analysis

st.set_page_config(page_title="Ex3 - Results", layout="wide")
st.title("Results - STFT Analysis of Music Recordings")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def seconds_to_mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def compute_hop_from_overlap(win_length: int, overlap_percent: int) -> int:
    hop = int(round(win_length * (1 - overlap_percent / 100)))
    return max(1, hop)


def make_heatmap(x, y, z, title, x_title, y_title, z_title=""):
    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorscale="Viridis",
            colorbar=dict(title=z_title),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=520,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def make_line_plot(x, y, title, x_title, y_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines"))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=320,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def band_summary_df(song_name: str, band_results: dict) -> pd.DataFrame:
    rows = []
    for label, values in band_results.items():
        rows.append(
            {
                "Song": song_name,
                "Band": label,
                "Mean relative band energy": round(values["EB_mean"], 4),
                "Number of frequency bins": len(values["indices"]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------
# Load files
# --------------------------------------------------
audio_files = helper.list_audio_files()

if not audio_files:
    st.info("No audio files found in `ex3/data/audio/`.")
    st.stop()

# --------------------------------------------------
# Controls
# --------------------------------------------------
st.markdown("## STFT settings")

col1, col2, col3 = st.columns(3)

with col1:
    selected_file = st.selectbox("Recording", audio_files)

with col2:
    window_name = st.selectbox(
        "Window function",
        options=["hann", "hamming", "blackman", "rectangular"],
        index=0,
    )

with col3:
    y_mode = st.selectbox(
        "Y-axis representation",
        options=["Magnitude", "Power", "dB", "Mel"],
        index=2,
    )

col4, col5, col6 = st.columns(3)

with col4:
    win_length = st.slider(
        "Window length (samples)",
        min_value=256,
        max_value=4096,
        step=256,
        value=2048,
    )

with col5:
    overlap_percent = st.slider(
        "Overlap (%)",
        min_value=0,
        max_value=90,
        step=10,
        value=50,
    )

with col6:
    n_mels = st.slider(
        "Mel bins (Mel mode only)",
        min_value=32,
        max_value=256,
        step=32,
        value=128,
    )

hop_length = compute_hop_from_overlap(win_length, overlap_percent)

# --------------------------------------------------
# Load selected audio
# --------------------------------------------------
audio = helper.load_audio(selected_file, mono=True)

if not audio:
    st.stop()

samples = audio["samples"]
sr = audio["sr"]
duration = audio["duration"]

delta_f = sr / win_length
delta_t_ms = 1000 * hop_length / sr

st.markdown(
    f"""
**Sample rate:** {sr} Hz  
**Duration:** {duration:.2f} s ({seconds_to_mmss(duration)})  
**Window:** {window_name}  
**Window length:** {win_length} samples  
**Hop length:** {hop_length} samples  
**Overlap:** {overlap_percent}%  
**Δf:** {delta_f:.2f} Hz  
**Δt:** {delta_t_ms:.2f} ms
"""
)

# --------------------------------------------------
# Audio player
# --------------------------------------------------
st.markdown("## Audio playback")

audio_path = helper.DATA_DIR / selected_file
audio_format = "audio/wav"
suffix = audio_path.suffix.lower()

if suffix == ".mp3":
    audio_format = "audio/mp3"
elif suffix == ".ogg":
    audio_format = "audio/ogg"
elif suffix == ".flac":
    audio_format = "audio/flac"
elif suffix == ".m4a":
    audio_format = "audio/mp4"
elif suffix == ".aiff":
    audio_format = "audio/aiff"

with open(audio_path, "rb") as f:
    st.audio(f.read(), format=audio_format)

# --------------------------------------------------
# Time segment
# --------------------------------------------------
st.markdown("## Time segment")

seg_col1, seg_col2 = st.columns(2)

with seg_col1:
    start_time = st.slider(
        "Start time (s)",
        min_value=0.0,
        max_value=float(duration),
        value=0.0,
        step=0.5,
    )

with seg_col2:
    end_time = st.slider(
        "End time (s)",
        min_value=0.0,
        max_value=float(duration),
        value=float(duration),
        step=0.5,
    )

if end_time <= start_time:
    st.error("End time must be greater than start time.")
    st.stop()

start_sample = int(start_time * sr)
end_sample = int(end_time * sr)
segment = samples[start_sample:end_sample]

if len(segment) < win_length:
    st.error("Selected time segment is shorter than the chosen window length.")
    st.stop()

# --------------------------------------------------
# STFT
# --------------------------------------------------
st.markdown("## Spectrogram")

stft_result = analysis.stft(
    segment,
    sr=sr,
    window_name=window_name,
    win_length=win_length,
    hop_length=hop_length,
)

times = stft_result["times"] + start_time
freqs = stft_result["freqs"]

if y_mode == "Magnitude":
    z = stft_result["magnitude"]
    y_axis = freqs
    y_title = "Frequency (Hz)"
    z_title = "Magnitude"
    spec_title = f"Magnitude spectrogram — {selected_file}"

elif y_mode == "Power":
    z = stft_result["power"]
    y_axis = freqs
    y_title = "Frequency (Hz)"
    z_title = "Power"
    spec_title = f"Power spectrogram — {selected_file}"

elif y_mode == "dB":
    z = analysis.to_db(
        stft_result["magnitude"],
        ref=np.max(stft_result["magnitude"]) + 1e-12
    )
    y_axis = freqs
    y_title = "Frequency (Hz)"
    z_title = "dB"
    spec_title = f"dB spectrogram — {selected_file}"

else:
    mel_result = analysis.to_mel(
        stft_result["power"],
        sr=sr,
        freqs=freqs,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sr / 2,
    )
    z = mel_result["mel_spec"]
    z = 10 * np.log10(np.maximum(z, 1e-12))
    y_axis = mel_result["mel_freqs"]
    y_title = "Mel frequency (Hz)"
    z_title = "Mel power (dB)"
    spec_title = f"Mel spectrogram — {selected_file}"

fig_spec = make_heatmap(
    x=times,
    y=y_axis,
    z=z,
    title=spec_title,
    x_title="Time (s)",
    y_title=y_title,
    z_title=z_title,
)

st.plotly_chart(fig_spec, width="stretch")

st.markdown(
    """
The spectrogram visualises how the spectral content changes over time. Horizontal structures
suggest more sustained harmonic components, while vertical bright stripes indicate transient
events such as drum hits or other abrupt changes in the signal.
"""
)

# --------------------------------------------------
# Spectrum overview
# --------------------------------------------------
st.markdown("## Average spectrum of the selected segment")

avg_mag = np.mean(stft_result["magnitude"], axis=1)

fig_avg_spec = go.Figure()
fig_avg_spec.add_trace(
    go.Scatter(
        x=freqs,
        y=avg_mag,
        mode="lines",
        name="Average magnitude spectrum",
    )
)
fig_avg_spec.update_layout(
    title="Average spectrum over the selected segment",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Average magnitude",
    height=350,
    margin=dict(l=60, r=20, t=60, b=60),
)
st.plotly_chart(fig_avg_spec, width="stretch")

# --------------------------------------------------
# Average mel spectrum
# --------------------------------------------------
st.markdown("## Average mel spectrum of the selected segment")

mel_avg_result = analysis.to_mel(
    stft_result["power"],
    sr=sr,
    freqs=freqs,
    n_mels=n_mels,
    fmin=0.0,
    fmax=sr / 2,
)

mel_spec = mel_avg_result["mel_spec"]
mel_freqs = mel_avg_result["mel_freqs"]

avg_mel = np.mean(mel_spec, axis=1)
avg_mel_db = 10 * np.log10(np.maximum(avg_mel, 1e-12))

fig_avg_mel = go.Figure()
fig_avg_mel.add_trace(
    go.Scatter(
        x=mel_freqs,
        y=avg_mel_db,
        mode="lines",
        name="Average mel spectrum",
    )
)
fig_avg_mel.update_layout(
    title="Average mel spectrum over the selected segment",
    xaxis_title="Mel frequency (Hz)",
    yaxis_title="Average mel power (dB)",
    height=350,
    margin=dict(l=60, r=20, t=60, b=60),
)
st.plotly_chart(fig_avg_mel, width="stretch")

st.markdown(
    """
The average mel spectrum summarises the spectral energy of the selected segment on a
perceptually motivated frequency scale. Compared with the linear frequency spectrum,
it emphasises how energy is distributed across frequency bands in a way that is closer
to human hearing.
"""
)
# --------------------------------------------------
# Band energy
# --------------------------------------------------
st.markdown("## Band energy features")

st.caption(
    "The band limits were chosen after inspecting the spectrograms and average spectra of the recordings. "
    "They separate bass-dominated, mid-range, upper-mid, and very high-frequency content."
)

bands = [
    (0, 250),
    (250, 2000),
    (2000, 8000),
    (8000, sr / 2),
]

band_results = analysis.compute_band_energy(stft_result, bands)
band_df = band_summary_df(selected_file, band_results)

st.dataframe(band_df, width="stretch", hide_index=True)

fig_band = go.Figure()

for label, values in band_results.items():
    fig_band.add_trace(
        go.Scatter(
            x=times,
            y=values["EB_rel"],
            mode="lines",
            name=label,
        )
    )

fig_band.update_layout(
    title="Relative band energy over time",
    xaxis_title="Time (s)",
    yaxis_title="Relative band energy",
    height=420,
    margin=dict(l=60, r=20, t=60, b=60),
)

st.plotly_chart(fig_band, width="stretch")

# --------------------------------------------------
# Additional metrics
# --------------------------------------------------
st.markdown("## Additional spectral metrics")

centroid = analysis.spectral_centroid(stft_result)
flatness = analysis.spectral_flatness(stft_result)

metric_df = pd.DataFrame(
    [
        {
            "Metric": "Mean spectral centroid (Hz)",
            "Value": round(float(np.mean(centroid)), 2),
        },
        {
            "Metric": "Std spectral centroid (Hz)",
            "Value": round(float(np.std(centroid)), 2),
        },
        {
            "Metric": "Mean spectral flatness",
            "Value": round(float(np.mean(flatness)), 4),
        },
        {
            "Metric": "Std spectral flatness",
            "Value": round(float(np.std(flatness)), 4),
        },
    ]
)

st.dataframe(metric_df, width="stretch", hide_index=True)

col_m1, col_m2 = st.columns(2)

with col_m1:
    fig_centroid = make_line_plot(
        x=times,
        y=centroid,
        title="Spectral centroid over time",
        x_title="Time (s)",
        y_title="Centroid (Hz)",
    )
    st.plotly_chart(fig_centroid, width="stretch")

with col_m2:
    fig_flatness = make_line_plot(
        x=times,
        y=flatness,
        title="Spectral flatness over time",
        x_title="Time (s)",
        y_title="Flatness",
    )
    st.plotly_chart(fig_flatness, width="stretch")

# --------------------------------------------------
# Comparison of both songs
# --------------------------------------------------
st.markdown("## Comparison of both recordings")

comparison_rows = []

for fname in audio_files:
    a = helper.load_audio(fname, mono=True)
    if not a:
        continue

    x = a["samples"]

    if len(x) < 2048:
        continue

    res = analysis.stft(
        x,
        sr=a["sr"],
        window_name="hann",
        win_length=2048,
        hop_length=1024,
    )

    band_res = analysis.compute_band_energy(
        res,
        [
            (0, 250),
            (250, 2000),
            (2000, 8000),
            (8000, a["sr"] / 2),
        ],
    )

    centroid_all = analysis.spectral_centroid(res)
    flatness_all = analysis.spectral_flatness(res)

    row = {
        "Recording": fname,
        "Mean centroid (Hz)": round(float(np.mean(centroid_all)), 2),
        "Mean flatness": round(float(np.mean(flatness_all)), 4),
    }

    for label, values in band_res.items():
        row[label] = round(values["EB_mean"], 4)

    comparison_rows.append(row)

if comparison_rows:
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, width="stretch", hide_index=True)

    # grouped bar plot for band energies
    band_labels = [c for c in comparison_df.columns if c not in ["Recording", "Mean centroid (Hz)", "Mean flatness"]]

    fig_cmp_bands = go.Figure()
    for _, row in comparison_df.iterrows():
        fig_cmp_bands.add_trace(
            go.Bar(
                x=band_labels,
                y=[row[label] for label in band_labels],
                name=row["Recording"],
            )
        )

    fig_cmp_bands.update_layout(
        title="Comparison of mean relative band energy",
        xaxis_title="Frequency band",
        yaxis_title="Mean relative band energy",
        barmode="group",
        height=420,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    st.plotly_chart(fig_cmp_bands, width="stretch")

    # grouped bar plot for centroid / flatness
    fig_cmp_metrics = go.Figure()
    for _, row in comparison_df.iterrows():
        fig_cmp_metrics.add_trace(
            go.Bar(
                x=["Mean centroid (Hz)", "Mean flatness"],
                y=[row["Mean centroid (Hz)"], row["Mean flatness"]],
                name=row["Recording"],
            )
        )

    fig_cmp_metrics.update_layout(
        title="Comparison of additional spectral metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        height=420,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    st.plotly_chart(fig_cmp_metrics, width="stretch")

st.divider()
st.caption("DSP Exercise 3 · FH Joanneum · 2026")