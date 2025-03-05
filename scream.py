import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore

# Define paths
positive_path = "Dataset/positive"
negative_path = "Dataset/negative"

# Function to extract features from an audio file
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # Extract 40 MFCC features
    return np.mean(mfccs.T, axis=0)

# Prepare the dataset
def prepare_dataset(folder_path, label):
    features, labels = [], []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):  # Ensure only .wav files are processed
            file_path = os.path.join(folder_path, file_name)
            try:
                features.append(extract_features(file_path))
                labels.append(label)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    return features, labels

# Load and prepare the dataset
positive_features, positive_labels = prepare_dataset(positive_path, 1)
negative_features, negative_labels = prepare_dataset(negative_path, 0)

# Combine features and labels
X = np.array(positive_features + negative_features)
y = np.array(positive_labels + negative_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid for binary classification
])



# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model in .h5 format
model.save("audio_classification_model.h5")

print("Model saved as 'audio_classification_model.h5'")
