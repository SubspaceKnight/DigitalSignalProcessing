"""
Run with:  uv run streamlit run Home.py
"""
import streamlit as st

st.set_page_config(
    page_title="DSP Exercise 1 - Time Domain Analysis",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏎️ DSP Exercise 1 - Time Domain Analysis")
st.markdown(
    """
    **Course:** Digital Signal Processing
    **Dataset:** Bahrain Grand Prix 2024 telemetry. Driver: Max Verstappen (VER)
    """
)

st.info("Use the **sidebar** to navigate between sections of the report.")

col1 = st.columns(1)

with col1:
    st.markdown(
        """
        ### About this report
        This app documents Exercise 1 of the DSP course.  
        We analyze Max Verstappen's onboard telemetry strictly in the **time domain** —
        no filtering, no Fourier transforms yet — to see how much information raw
        signals already reveal about driving behavior, consistency, and lap-to-lap
        variability.

        Navigate using the sidebar:
        - **Introduction** - signal theory recap
        - **Methods** - what we did and why
        - **Results** - interactive plots
        - **Discussion** - what we found and what's missing
        - **Race Visualisation** - displaying the laps on a track map (by X,Y coordinates)
        """
    )

# with col2:
#     st.markdown("### A lap at Bahrain")
#     # The YouTube Short the professor requested
#     st.video("https://www.youtube.com/shorts/8E36bKfg_qU")