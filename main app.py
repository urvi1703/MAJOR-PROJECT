import streamlit as st
import numpy as np
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
from utils import extract_features, plot_spectrogram

st.title("🎤 Drone Detection System")

model = tf.keras.models.load_model("models/drone_detection_model.h5")

# File Upload
uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio("temp_audio.wav", format="audio/wav")

    features, _ = extract_features("temp_audio.wav")[0]
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)
    predicted_label = "Drone" if np.argmax(prediction) == 1 else "Background Noise"

    st.write(f"### 🔍 Prediction: {predicted_label}")
    plot_spectrogram("temp_audio.wav")

# Real-Time Recording
if st.button("🎙 Record & Detect"):
    duration = 3  # seconds
    samplerate = 22050
    st.write("Recording... Speak now!")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()

    sf.write("realtime_audio.wav", audio, samplerate)
    st.write("Recording saved! Processing...")

    features, _ = extract_features("realtime_audio.wav")[0]
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)
    predicted_label = "Drone" if np.argmax(prediction) == 1 else "Background Noise"

    st.write(f"### 🔍 Prediction: {predicted_label}")
    plot_spectrogram("realtime_audio.wav")
