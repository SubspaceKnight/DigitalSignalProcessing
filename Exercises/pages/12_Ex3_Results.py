import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import Exercises.ex3.helper   as helper
import Exercises.ex3.analysis as analysis

st.set_page_config(page_title="Ex3 - Results", layout="wide")
st.title("Results - STFT Analysis of Music Recordings")

#Sidebar controls
st.sidebar.header("STFT settings")

audio_files = helper.list_audio_files()

if not audio_files:
    st.warning(
        "No audio files found in `data/audio/`. "
        "Add `.wav` or `.mp3` files there and reload."
    )
    st.stop()

selected_file = st.sidebar.selectbox("Recording", options=audio_files)

window_name = st.sidebar.selectbox(
    "Window function",
    options=analysis.WINDOW_FUNCTIONS,
    index=0,
)
win_length = st.sidebar.select_slider(
    "Window length (samples)",
    options=[256, 512, 1024, 2048, 4096],
    value=1024,
)
overlap_pct = st.sidebar.slider("Overlap (%)", min_value=0, max_value=90, value=50, step=10)
hop_length  = max(1, int(win_length * (1 - overlap_pct / 100)))

st.sidebar.markdown(
    f"**Δf** = {0:.0f} Hz  *(fill when fs known)*  \n"
    f"**Δt** = {hop_length} samples"
)

st.sidebar.header("Spectrogram view")
repr_mode = st.sidebar.radio(
    "Y-axis representation",
    options=["Magnitude", "Power", "dB", "Mel"],
    index=2,
)
n_mels = st.sidebar.slider("Mel bins (Mel mode only)", 32, 256, 128, 32)

st.sidebar.header("Time segment")


#Load audio
@st.cache_data
def load(fname):
    return helper.load_audio(fname)

audio = load(selected_file)
if not audio:
    st.stop()

info = helper.audio_info_df(audio)
c1, c2, c3, c4, c5 = st.columns(5)
for col, (k, v) in zip([c1, c2, c3, c4, c5], list(info.items())[:5]):
    col.metric(k, v)

t_start = st.sidebar.slider(
    "Start (s)", 0.0, max(0.0, audio["duration"] - 1.0), 0.0, 0.1
)
t_end   = st.sidebar.slider(
    "End (s)", 1.0, audio["duration"], min(30.0, audio["duration"]), 0.1
)

#Compute STFT
sr  = audio["sr"]
seg = audio["samples"][int(t_start * sr) : int(t_end * sr)]

@st.cache_data
def compute_stft(fname, t0, t1, wname, wlen, hlen):
    a   = helper.load_audio(fname)
    seg = a["samples"][int(t0 * a["sr"]) : int(t1 * a["sr"])]
    return analysis.stft(seg, a["sr"], window_name=wname,
                         win_length=wlen, hop_length=hlen)

stft_result = compute_stft(
    selected_file, t_start, t_end, window_name, win_length, hop_length
)

st.info(
    f"Segment: {t_start:.1f} - {t_end:.1f} s · "
    f"Frames: {stft_result['n_frames']} · "
    f"Freq bins: {stft_result['n_freqs']} · "
    f"Δf ≈ {sr / win_length:.2f} Hz · "
    f"Δt ≈ {hop_length / sr * 1000:.1f} ms"
)

#Tabs
tab_spec, tab_bands, tab_metrics = st.tabs(["Spectrogram", "Band energy", "Additional metrics"])

#TAB 1: Spectrogram
with tab_spec:
    st.subheader(f"Spectrogram — {repr_mode} · {selected_file}")

    times = stft_result["times"] + t_start
    freqs = stft_result["freqs"]
    mag = stft_result["magnitude"]
    pwr = stft_result["power"]

    if repr_mode == "Magnitude":
        z = mag
        z_label = "Magnitude"
        colorscale = "Viridis"
        zmin = None 
        zmax = None
    elif repr_mode == "Power":
        z = pwr
        z_label = "Power"
        colorscale = "Viridis"
        zmin = None
        zmax = None
    elif repr_mode == "dB":
        z = analysis.to_db(mag)
        z_label = "Magnitude (dB)"
        colorscale = "Magma"
        zmin = z.max() - 80
        zmax = z.max()
    else:  # Mel
        mel_out = analysis.to_mel(pwr, sr, freqs, n_mels=n_mels)
        z = analysis.to_db(np.sqrt(mel_out["mel_spec"]))
        freqs = mel_out["mel_freqs"]
        z_label = "Mel power (dB)"
        colorscale = "Magma"
        zmin = z.max() - 80
        zmax = z.max()

    fig_spec = go.Figure(go.Heatmap(
        x=times,
        y=freqs,
        z=z,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=z_label),
    ))
    fig_spec.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)" if repr_mode != "Mel" else "Mel frequency (Hz)",
        title=(
            f"{repr_mode} spectrogram · window = {window_name} · "
            f"L = {win_length} · hop = {hop_length}"
        ),
        height=420,
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="closest",
    )
    st.plotly_chart(fig_spec, use_container_width=True)
    st.caption(
        f"Window: {window_name} · Length: {win_length} samples "
        f"({win_length / sr * 1000:.1f} ms) · "
        f"Overlap: {overlap_pct}% · Hop: {hop_length} samples "
        f"({hop_length / sr * 1000:.1f} ms)"
    )

    # TODO: add a second recording for side-by-side comparison once chosen


#TAB 2: Band energy
with tab_bands:
    st.subheader("Band energy features")

    #TODO: adjust band boundaries to suit our recordings' frequency range
    nyq = sr // 2
    default_bands = [(0, 250), (250, 2000), (2000, 8000), (8000, nyq)]

    st.markdown(
        "Default bands (adjust in code once our recordings are analysed):"
    )
    band_results = analysis.compute_band_energy(stft_result, default_bands)
    summary_df   = analysis.band_energy_summary_df(band_results)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    #Time-series of relative band energy
    fig_bands = go.Figure()
    colors_b = ["cornflowerblue", "coral", "mediumseagreen", "goldenrod"]
    for (label, res), color in zip(band_results.items(), colors_b):
        fig_bands.add_trace(go.Scatter(
            x=stft_result["times"] + t_start,
            y=res["EB_rel"],
            mode="lines",
            line=dict(color=color, width=1.2),
            name=label,
        ))
    fig_bands.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Relative band energy",
        title="Relative band energy over time",
        height=340,
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig_bands, use_container_width=True)

    #TODO: add comparison across recordings once both songs are loaded


#TAB 3: Additional metrics
with tab_metrics:
    st.subheader("Additional metrics")

    sc = analysis.spectral_centroid(stft_result)
    sf = analysis.spectral_flatness(stft_result)
    t_ax = stft_result["times"] + t_start

    fig_m = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Spectral centroid (Hz)", "Spectral flatness"),
    )
    fig_m.add_trace(go.Scatter(
        x=t_ax, y=sc, mode="lines",
        line=dict(color="cornflowerblue", width=1.2), name="Centroid",
    ), row=1, col=1)
    fig_m.add_trace(go.Scatter(
        x=t_ax, y=sf, mode="lines",
        line=dict(color="coral", width=1.2), name="Flatness",
    ), row=2, col=1)
    fig_m.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig_m.update_yaxes(title_text="Hz",   row=1, col=1)
    fig_m.update_yaxes(title_text="0-1",  row=2, col=1)
    fig_m.update_layout(
        height=440,
        margin=dict(l=60, r=20, t=60, b=60),
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(fig_m, use_container_width=True)
    st.caption(
        "Spectral centroid: high = bright/noisy timbre · low = bass-heavy.  "
        "Spectral flatness: near 1 = noise-like · near 0 = tonal."
    )

    #TODO: summary table comparing both recordings once available

st.divider()
st.caption("DSP Exercise 3 * FH Joanneum * 2026")
