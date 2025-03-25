import librosa
import numpy as np
import os
import pickle

def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

def preprocess_dataset(dataset_path, output_file):
    data = []
    labels = []
    for label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(class_path, file)
                    features = extract_mfcc(file_path)
                    data.append(features)
                    labels.append(label)
    with open(output_file, "wb") as f:
        pickle.dump((np.array(data), np.array(labels)), f)

if __name__ == "__main__":
    preprocess_dataset("/content/drive/MyDrive/dataset", "dataset.pkl")
