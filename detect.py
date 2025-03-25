import librosa
import numpy as np
import tensorflow as tf
import pickle

def classify_audio(audio_path, model_path="drone_model.h5", encoder_path="dataset.pkl", segment_duration=1):
    model = tf.keras.models.load_model(model_path)
    with open(encoder_path, "rb") as f:
        _, labels, _ = pickle.load(f)
    
    y, sr = librosa.load(audio_path, sr=22050)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    segment_samples = segment_duration * sr
    predictions = []
    
    for start in range(0, len(y), segment_samples):
        segment = y[start:start + segment_samples]
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        mfcc = np.expand_dims(mfcc, axis=(0, -1))
        prediction = model.predict(mfcc)
        predictions.append(np.argmax(prediction))
    
    final_prediction = max(set(predictions), key=predictions.count)
    return labels[final_prediction]

if __name__ == "__main__":
    print(classify_audio("sample.wav"))
