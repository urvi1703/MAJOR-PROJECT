import librosa
import numpy as np
import os
import pickle
import librosa.display
import matplotlib.pyplot as plt

def extract_features(audio_path, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    spectrogram = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
    return np.mean(mfcc.T, axis=0), spectrogram

def preprocess_dataset(dataset_path, output_file):
    data = []
    labels = []
    spectrograms = []
    for label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(class_path, file)
                    mfcc, spectrogram = extract_features(file_path)
                    data.append(mfcc)
                    spectrograms.append(spectrogram)
                    labels.append(label)
    with open(output_file, "wb") as f:
        pickle.dump((np.array(data), np.array(labels), np.array(spectrograms)), f)
        
if __name__ == "__main__":
    preprocess_dataset("https://drive.google.com/drive/folders/1NGMg21f-WxpzgHWjM6OlAPV1ol1DcQY7?usp=drive_link", "dataset.pkl")
