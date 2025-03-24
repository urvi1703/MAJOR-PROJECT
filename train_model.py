# train_model.py (Training the CNN Model)
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

from google.colab import drive
drive.mount('/content/drive')

dataset_path = "https://drive.google.com/drive/folders/1NGMg21f-WxpzgHWjM6OlAPV1ol1DcQY7?usp=sharing"

def load_data(dataset_path):
    X, Y = [], []
    labels = {"drone": 0, "background": 1}
    
    for label in labels:
        folder_path = os.path.join(dataset_path, label)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            signal, sr = librosa.load(file_path, sr=22050)
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
            mfccs = np.resize(mfccs, (40, 32, 1))  # Reshaping
            X.append(mfccs)
            Y.append(labels[label])
    
    return np.array(X), to_categorical(np.array(Y))

# Load dataset
X, Y = load_data(dataset_path)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, 32, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=32)

# Save Model & Training History
model.save("models/drone_cnn_model.h5")
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Model training complete and saved!")
