import streamlit as st
import librosa
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from detect import classify_audio

st.title("Drone Detection using Acoustic Signatures")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
record_button = st.button("Record & Analyze Live Audio")

def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.show()

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name
    
    st.audio(temp_path, format='audio/wav')
    st.write("Analyzing...")
    prediction = classify_audio(temp_path)
    st.write(f"Prediction: {prediction}")
    st.pyplot(plot_spectrogram(temp_path))
