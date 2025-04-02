import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def extract_features(audio_path, n_mfcc=40, segment_length=1):
    """
    Extracts MFCC features and spectrogram from an audio file.
    Splits long audio into 1-second segments and processes each.
    """
    y, sr = librosa.load(audio_path, sr=16000)
    segments = []
    
    for i in range(0, len(y), sr * segment_length):
        segment = y[i:i + sr * segment_length]
        if len(segment) < sr * segment_length:
            break
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        spectrogram = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=segment, sr=sr), ref=np.max)
        segments.append((np.mean(mfcc.T, axis=0), spectrogram))
    
    return segments

def plot_spectrogram(audio_path):
    """
    Generates and displays a spectrogram of an audio file.
    """
    y, sr = librosa.load(audio_path, sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    st.pyplot(fig)
