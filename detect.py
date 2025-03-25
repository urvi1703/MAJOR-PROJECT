import librosa
import numpy as np
import tensorflow as tf
import pickle

def classify_audio(audio_path, model_path="drone_model.h5", encoder_path="dataset.pkl"):
    model = tf.keras.models.load_model(model_path)
    with open(encoder_path, "rb") as f:
        _, labels = pickle.load(f)
    
    features = extract_mfcc(audio_path)
    features = np.expand_dims(features, axis=0)
    
    prediction = model.predict(features)
    class_idx = np.argmax(prediction)
    return labels[class_idx]

if __name__ == "__main__":
    print(classify_audio("sample.wav"))
