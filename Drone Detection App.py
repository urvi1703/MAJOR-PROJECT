import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
import tempfile
from utils import extract_mfcc, plot_spectrogram

# Load the trained model
MODEL_PATH = "drone_cnn_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def predict_audio(audio_file):
    mfccs = extract_mfcc(audio_file)
    if mfccs is None:
        return "Error processing audio"
    
    mfccs = mfccs.reshape(1, 40, 32, 1)
    prediction = model.predict(mfccs)
    label = "Drone" if prediction > 0.5 else "Background Noise"
    return label

st.title("🚁 Drone Detection Using Acoustic Signatures")

option = st.radio("Choose an input method:", ("Upload Audio File", "Record Using Microphone"))

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name
        
        st.audio(audio_path, format="audio/wav")
        result = predict_audio(audio_path)
        st.write("Prediction:", result)
        plot_spectrogram(audio_path)

elif option == "Record Using Microphone":
    duration = st.slider("Select duration (seconds)", 1, 5, 3)
    if st.button("Record Audio"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            st.write("Recording...")
            recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
            sd.wait()
            sf.write(tmp.name, recording, 16000)
            audio_path = tmp.name
            
            st.write("Recorded Audio:")
            st.audio(audio_path, format="audio/wav")
            result = predict_audio(audio_path)
            st.write("Prediction:", result)
            plot_spectrogram(audio_path)
