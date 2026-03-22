"""
Run with:  uv run streamlit run Main_Page.py
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

st.set_page_config( page_title="DSP Exercises", layout="wide", )
st.title("Digital Signal Processing - FH Joanneum")

# st.markdown(
#     """
#     **Course:** Digital Signal Processing
#     **Dataset:** Bahrain Grand Prix 2024 telemetry. Driver: Max Verstappen (VER)
#     """
# )

st.info("Use the **sidebar** to navigate between sections of the report.")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        ### Exercise 1 - Time Domain Analysis
        Analysis of Max Verstappen's telemetry from the 2024 Bahrain Grand Prix.
        Purely time-domain: lap consistency, stint structure, braking zones.
        """
    )
    if st.button("Go to Exercise 1"):
        st.switch_page("pages/1_Ex1_Introduction.py")

with col2:
    st.markdown(
        """
        ### Exercise 2 - Frequency Domain Analysis
        *Coming soon*
        """
    )
    if st.button("Go to Exercise 2"):
        st.switch_page("pages/6_Ex2_Introduction.py")

# with col1:
#     st.markdown(
#         """
#         ### About this report
#         This app documents Exercise 1 of the DSP course.  
#         We analyze Max Verstappen's onboard telemetry strictly in the **time domain** -
#         no filtering, no Fourier transforms yet - to see how much information raw
#         signals already reveal about driving behavior, consistency, and lap-to-lap
#         variability.

#         Navigate using the sidebar:
#         - **Introduction** - signal theory recap
#         - **Methods** - what we did and why
#         - **Results** - interactive plots
#         - **Discussion** - what we found and what's missing
#         - **Race Visualisation** - displaying the laps on a track map (by X,Y coordinates)
#         """
#     )

# with col2:
#     st.markdown("### A lap at Bahrain")
#     # The YouTube Short the professor requested
#     st.video("https://www.youtube.com/shorts/8E36bKfg_qU")