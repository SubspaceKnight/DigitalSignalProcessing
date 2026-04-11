import streamlit as st

st.set_page_config(page_title="Ex3 - Discussion", layout="wide")
st.title("Discussion - STFT Results & Interpretation")

st.markdown(
    r"""
    ## Differences across recordings

    The two recordings show clear differences in their time-frequency structure.
    *Lollipop* displays a more regular and repetitive pattern over time, with many
    evenly spaced vertical structures that suggest a stable beat and recurring
    rhythmic events. The overall spectrogram appears more uniform, which is consistent
    with a more repetitive, beat-oriented musical structure.

    *Chop Suey!* shows stronger temporal variation and a less uniform spectrogram.
    There are more abrupt changes across sections of the song, which is consistent
    with stronger dynamic contrasts and denser instrumentation. Compared with
    *Lollipop*, the spectral content varies more strongly over time, and bright
    vertical structures appear less regular but more section-dependent.

    In both recordings, horizontal structures indicate sustained harmonic content,
    while vertical bright stripes correspond to transient events. However, these
    patterns are more regular in *Lollipop* and more variable in *Chop Suey!*.

    ## Y-axis representation — what each reveals

    ### Magnitude
    Magnitude directly shows the amplitude of each frequency component. It is easy
    to interpret, but weaker components can disappear next to stronger ones because
    the dynamic range is large.

    ### Power
    Power emphasises dominant components even more strongly because it squares the
    magnitude. This is useful when energy distribution is more important than raw
    amplitude.

    ### Decibel scale
    The decibel representation was the most useful view for our recordings. It made
    both strong and weak spectral structures visible at the same time and therefore
    gave the clearest overview of the differences between the songs. In particular,
    it helped reveal the recurring spectral structure in *Lollipop* and the stronger
    section-to-section variation in *Chop Suey!*.

    ### Mel scale
    The Mel representation compresses the upper frequency range and highlights the
    perceptually important low- and mid-frequency structure. It would be especially
    useful as input for machine learning, but for visual interpretation of our two
    songs the dB spectrogram was more informative.

    ## The time-frequency trade-off

    The window length $L$ governs a fundamental trade-off between time and frequency
    resolution. A longer window improves frequency resolution but reduces time
    precision, while a shorter window improves time precision but makes neighbouring
    frequencies harder to separate.

    In our analysis, we used a window length of 2048 samples and a hop length of
    1024 samples with 50 % overlap. This was a good compromise for both recordings.
    It was long enough to represent harmonic structure clearly, while still preserving
    enough temporal detail to show transient and rhythmic events. For *Lollipop*,
    this setting captured the repetitive beat structure well. For *Chop Suey!*,
    it was sufficient to reveal the larger structural and spectral changes across
    sections.

    ## Effect of window functions

    The Hann window proved to be an appropriate default choice for our analysis.
    Compared with the rectangular window, it reduces spectral leakage and produces
    a cleaner and more interpretable spectrogram. This is especially important in
    music, where many neighbouring frequency components occur simultaneously.

    For our recordings, the Hann window provided a good balance between side-lobe
    suppression and frequency localisation. A rectangular window would have produced
    stronger leakage, while a Blackman window would have reduced leakage further
    but at the cost of broader peaks and reduced sharpness.

    ## Band energy results — interpretation

    The band energy comparison shows that both songs are dominated by low-frequency
    and low-mid energy, but there are still meaningful differences.

    | Band | Lollipop $\bar{E}_B$ | Chop Suey! $\bar{E}_B$ | Interpretation |
    |---|---:|---:|---|
    | 0–250 Hz | 0.5161 | 0.4950 | *Lollipop* is slightly more bass-dominated |
    | 250–2000 Hz | 0.3196 | 0.3258 | Very similar mid-range contribution |
    | 2000–8000 Hz | 0.1591 | 0.1559 | Similar presence/harmonic energy |
    | 8000–Nyquist Hz | 0.0051 | 0.0130 | *Chop Suey!* has clearly more very high-frequency energy |

    These results fit the visual impression of the spectrograms. *Lollipop* appears
    slightly more dominated by bass and regular rhythmic structure, while *Chop Suey!*
    contains more energy in the upper frequency range, which is consistent with a
    brighter and more aggressive sound character.

    ## Spectral centroid — interpretation

    The mean spectral centroid is higher for *Chop Suey!* (2930.7 Hz) than for
    *Lollipop* (2557.29 Hz). This indicates that *Chop Suey!* has a brighter overall
    timbre and more high-frequency content on average.

    The centroid over time also varies strongly in both songs, but the higher mean
    value of *Chop Suey!* matches the expectation from the spectrogram and from the
    musical style. This supports the interpretation that *Chop Suey!* is spectrally
    brighter than the more beat-oriented *Lollipop*.

    ## Spectral flatness — interpretation

    The mean spectral flatness is also higher for *Chop Suey!* (0.0055) than for
    *Lollipop* (0.0009). Although both values are low overall, the difference is
    still meaningful. It suggests that *Chop Suey!* contains more noise-like or
    broadband components, while *Lollipop* is more spectrally concentrated and tonal.

    This is consistent with the character of the songs: *Lollipop* appears more
    repetitive and structured, whereas *Chop Suey!* contains denser textures and
    more irregular high-frequency content.

    ## If we had to train a classifier

    If the goal were to classify these recordings automatically, the most useful
    features for this particular pair would be the Mel spectrogram, band energy,
    spectral centroid, and spectral flatness.

    The Mel spectrogram would be the strongest general-purpose input because it
    preserves time-frequency structure in a perceptually meaningful form. Band
    energy would provide a compact summary of how energy is distributed across
    bass, mid, and treble regions. Spectral centroid would capture overall brightness,
    and spectral flatness would help distinguish more tonal from more noise-like
    passages.

    For a classical machine learning pipeline, the mean and standard deviation of
    these frame-wise features would form a compact and informative feature vector.
    For a deep learning approach, the raw Mel spectrogram would likely be the most
    suitable representation.
    """
)

st.divider()
st.caption("DSP Exercise 3 · FH Joanneum · 2026")