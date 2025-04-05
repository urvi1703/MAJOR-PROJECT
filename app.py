import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import soundfile as sf

# Streamlit Title
st.title("🚁 Drone Detection using Audio Classification")

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("drone_cnn_model.h5")
    return model

model = load_model()

# ✔️ Correct Feature Extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Pad or truncate to ensure shape (40, 32)
    if mfcc.shape[1] < 32:
        pad_width = 32 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :32]

    # Final reshape: (1, 40, 32, 1)
    mfcc = mfcc.reshape(1, 40, 32, 1)
    return mfcc

# ✔️ Visualization
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

# ✔️ Prediction Logic
def predict_and_display(file_path):
    features = extract_features(file_path)
    st.write("✅ Feature shape:", features.shape)

    prediction = model.predict(features)

    if prediction.shape[1] == 2:
        label_map = ["Background Noise", "Drone"]
        predicted_index = np.argmax(prediction)
        label = label_map[predicted_index]
        confidence = prediction[0][predicted_index]
    else:
        confidence = prediction[0][0]
        label = "Drone" if confidence > 0.5 else "Background Noise"
        confidence = confidence if confidence > 0.5 else 1 - confidence

    st.success(f"🧠 Prediction: **{label}** (Confidence: {confidence*100:.2f}%)")

# ✔️ File Upload
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    file_path = "temp.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    show_visuals(file_path)
    predict_and_display(file_path)
else:
    st.info("Waiting for file upload...")

# ✔️ Optional: Record from Microphone
if st.checkbox("Use Microphone (local only)"):
    try:
        import sounddevice as sd

        duration = 5
        fs = 16000

        st.write("🎙️ Click the button to record")
        if st.button("Record Now"):
            st.write("Recording...")
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            sf.write("mic_input.wav", audio, fs)
            st.write("Recording complete")

            show_visuals("mic_input.wav")
            predict_and_display("mic_input.wav")
    except Exception as e:
        st.warning(f"Microphone access failed: {e}")
