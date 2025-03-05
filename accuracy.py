import os
import librosa
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model # type: ignore

# Load the saved model
model = load_model("Website/audio_classification_model.h5")

# Function to extract features from audio files
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Prepare dataset for testing
def prepare_test_data(folder_path, label):
    features = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            try:
                features.append(extract_features(file_path))
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return features, labels

# Load test data
positive_test_path = "Dataset/positive"  # Path to positive test samples
negative_test_path = "Dataset/negative"  # Path to negative test samples

positive_features, positive_labels = prepare_test_data(positive_test_path, 1)
negative_features, negative_labels = prepare_test_data(negative_test_path, 0)

# Combine test features and labels
X_test = np.array(positive_features + negative_features)
y_test = np.array(positive_labels + negative_labels)

# Predict on the test dataset
y_pred_prob = model.predict(X_test)  # Predictions as probabilities
y_pred = (y_pred_prob > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
