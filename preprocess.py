import os
import numpy as np
import librosa
import librosa.display
import pickle
import zipfile
import gdown
import matplotlib.pyplot as plt

# Function to download dataset from Google Drive
def download_dataset(drive_file_id, output_file="dataset.zip"):
    url = f"https://drive.google.com/uc?id={drive_file_id}"
    gdown.download(url, output_file, quiet=False)
    print(f"✅ Dataset downloaded as {output_file}")

# Function to extract dataset
def extract_dataset(zip_path, extract_to="data"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Dataset extracted to {extract_to}")

# Feature extraction function
def extract_features(audio_path, n_mfcc=40):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        spectrogram = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
        return np.mean(mfcc.T, axis=0), spectrogram
    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {e}")
        return None, None

# Preprocessing function
def preprocess_dataset(dataset_path, output_file="dataset.pkl"):
    data, labels, spectrograms = [], [], []
    
    for label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(class_path, file)
                    mfcc, spectrogram = extract_features(file_path)
                    if mfcc is not None:
                        data.append(mfcc)
                        spectrograms.append(spectrogram)
                        labels.append(label)

    # Save processed dataset
    with open(output_file, "wb") as f:
        pickle.dump((np.array(data), np.array(labels), np.array(spectrograms)), f)
    
    print(f"✅ Dataset preprocessed and saved as {output_file}")

# Plot a sample spectrogram
def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.show()

# Run script
if __name__ == "__main__":
    DRIVE_FILE_ID = "https://drive.google.com/drive/folders/1NGMg21f-WxpzgHWjM6OlAPV1ol1DcQY7?usp=drive_link"  # Replace with actual Google Drive File ID

    # Step 1: Download & Extract Dataset
    download_dataset(DRIVE_FILE_ID)
    extract_dataset("dataset.zip", "data")

    # Step 2: Preprocess Data
    preprocess_dataset("data", "dataset.pkl")
