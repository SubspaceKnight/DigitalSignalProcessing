import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

#keep in ram
@st.cache_data
def load_eeg_excel(filepath: str) -> dict:
    path = Path(filepath)
    if not path.exists():
        st.error(f"Excel file not found: {path}")
        return None
        
    try:
        df = pd.read_excel(path)
        
        #columns as NumPy arrays
        time = df['Time'].to_numpy()
        signal = df['Avg_EEG'].to_numpy()
        
        #our sampling frequency
        fs = 512 
        
        return {
            "time": time,
            "signal": signal,
            "fs": fs,
            "n_samples": len(signal)
        }
        
    except Exception as e:
        st.error(f"Error parsing the Excel file: {e}")
        return None