import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

def extract_features(audio_path, n_mfcc=40, segment_length=1):
    y, sr = librosa.load(audio_path, sr=22050)
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
    y, sr = librosa.load(audio_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.show()

