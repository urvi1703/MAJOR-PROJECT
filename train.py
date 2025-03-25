import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# Load Data
with open("dataset.pkl", "rb") as f:
    X, y, _ = pickle.load(f)

# Encode Labels
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Reshape Input
X = np.expand_dims(X, axis=-1)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN Model
model = keras.Sequential([
    keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(set(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save Model
model.save("drone_model.h5")
