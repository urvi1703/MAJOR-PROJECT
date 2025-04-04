import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import soundfile as sf

# Title
st.title("🚁 Drone Detection using Audio Classification")

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("drone_cnn_model.h5")  # Put your actual model name
    return model

model = load_model()

# Preprocessing function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed.reshape(1, -1)

# Visualization function
def show_visuals(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    
    st.subheader("Waveform:")
    fig1, ax1 = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Spectrogram:")
    fig2, ax2 = plt.subplots()
    stft = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(stft, sr=sr, x_axis='time', y_axis='log', ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    st.pyplot(fig2)

# File uploader
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    file_path = os.path.join("temp.wav")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    show_visuals(file_path)
    features = extract_features(file_path)
    prediction = model.predict(features)

    label = "Drone" if prediction[0][0] > 0.5 else "Background Noise"
    st.success(f"🧠 Prediction: **{label}** (Confidence: {prediction[0][0]:.2f})")

else:
    st.info("Waiting for file upload...")

# Optional: Audio Recorder (only works on local machines, not Codespaces)
if st.checkbox("Use Microphone (local only)"):
    try:
        import sounddevice as sd

        duration = 5  # seconds
        fs = 16000

        st.write("🎙️ Click the button to record")
        if st.button("Record Now"):
            st.write("Recording...")
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            sf.write("mic_input.wav", audio, fs)
            st.write("Recording complete")

            show_visuals("mic_input.wav")
            features = extract_features("mic_input.wav")
            prediction = model.predict(features)
            label = "Drone" if prediction[0][0] > 0.5 else "Background Noise"
            st.success(f"🧠 Prediction: **{label}** (Confidence: {prediction[0][0]:.2f})")
    except Exception as e:
        st.warning(f"Microphone access failed: {e}")
