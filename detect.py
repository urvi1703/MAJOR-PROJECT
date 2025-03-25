import librosa
import numpy as np
import tensorflow as tf
import pickle

def classify_audio(audio_path, model_path="drone_model.h5", encoder_path="dataset.pkl"):
    model = tf.keras.models.load_model(model_path)
    with open(encoder_path, "rb") as f:
        _, labels, _ = pickle.load(f)
    
    mfcc, _ = extract_features(audio_path)
    mfcc = np.expand_dims(mfcc, axis=(0, -1))
    
    prediction = model.predict(mfcc)
    class_idx = np.argmax(prediction)
    return labels[class_idx]

if __name__ == "__main__":
    print(classify_audio("sample.wav"))
