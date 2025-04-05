import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

# Streamlit title
st.title("🚁 Drone Detection using Audio Classification")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("drone_cnn_model.h5")

model = load_model()

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < 32:
        pad_width = 32 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :32]

    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Shape: (1, 40, 32, 1)
    return mfcc

# Visualization
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

# Prediction
def predict_and_display(file_path):
    features = extract_features(file_path)
    st.write(f"✅ Extracted features shape: {features.shape}")

    try:
        prediction = model.predict(features)
        st.write(f"📊 Model raw prediction: {prediction}")

        # Handle binary or categorical output
        if prediction.shape[1] == 2:
            label_map = ["Background Noise", "Drone"]
            predicted_index = np.argmax(prediction)
            label = label_map[predicted_index]
            confidence = prediction[0][predicted_index]
        else:
            confidence = prediction[0][0]
            label = "Drone" if confidence > 0.5 else "Background Noise"
            confidence = confidence if confidence > 0.5 else 1 - confidence

        st.success(f"🧠 Prediction: **{label}** (Confidence: {confidence * 100:.2f}%)")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# File uploader
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    file_path = "temp.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    show_visuals(file_path)
    predict_and_display(file_path)
else:
    st.info("Please upload a .wav audio file to begin.")
