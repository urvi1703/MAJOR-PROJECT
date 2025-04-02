import streamlit as st
import numpy as np
import tensorflow as tf
pip install sounddevice
import sounddevice as sd
import soundfile as sf
from utils import extract_features, plot_spectrogram
import pyaudio as pa

st.title("🎤 Drone Detection System")
import tensorflow as tf
model = tf.keras.models.load_model("drone_cnn_model.h5")  # Remove "models/" if not in a folder

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
    samplerate = 16000
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


import sounddevice as sd
import numpy as np
import soundfile as sf

# Record audio
duration = 5  # seconds
samplerate = 16000
print("Recording...")
audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
sd.wait()
sf.write("output.wav", audio, samplerate)
print("Recording complete.")


