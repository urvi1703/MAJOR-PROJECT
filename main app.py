import streamlit as st
import numpy as np
import tensorflow as tf
#import sounddevice as sd
import soundfile as sf
from utils import extract_features, plot_spectrogram

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

import soundfile as sf

data, samplerate = sf.read("audio_file.wav")

# Open stream for recording
stream = p.open(format=sf.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024)

print("Recording...")

frames = []

# Record audio for 5 seconds
for _ in range(0, int(44100 / 1024 * 5)):
    data = stream.read(1024)
    frames.append(data)

print("Finished recording.")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Save the audio to a file
with wave.open("output.wav", 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))

