### Streamlit UI ###
import streamlit as st
import librosa
import numpy as np
import tempfile
from detect import classify_audio

st.title("Drone Detection using Acoustic Signatures")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name
    prediction = classify_audio(temp_path)
    st.write(f"Prediction: {prediction}")
