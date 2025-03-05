import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Load the saved model
model = load_model("audio_classification_model.h5")

# Function to extract features from an audio file
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # Extract 40 MFCC features
    return np.mean(mfccs.T, axis=0)

# Test function to predict new audio files
def predict_audio(file_path):
    try:
        features = extract_features(file_path)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        prediction = model.predict(features)
        label = "Positive" if prediction[0][0] > 0.5 else "Negative"
        print(f"File: {file_path}, Prediction: {label}, Confidence: {prediction[0][0]:.2f}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# List of test audio files
test_files = [
    "test/test5p.wav",
    "test/test2p.wav",
    "test/test3n.wav",
    "test/test4n.wav"
]

# Predict and print results for each file
for file in test_files:
    predict_audio(file)
