import os

import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def extract_features(audio_folder):
    """
    Extracts Spectral Centroid, Zero-Crossing Rate, and Spectral Flux for each audio file.
    Assigns labels based on folder: 0 for 'Percussion', 1 for others.
    """
    features = []
    labels = []
    file_names = []

    for folder in audio_folder:
        label = 1 if 'Chords' in folder else 0
        audio_files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3'))]

        for audio_file in audio_files:
            audio_path = os.path.join(folder, audio_file)
            try:
                # Load the audio file
                y, sr = librosa.load(audio_path, sr=None)

                # Calculate Spectral Centroid
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

                # Calculate Zero-Crossing Rate
                zcr = librosa.feature.zero_crossing_rate(y)[0]

                # Calculate Spectral Flux
                S = np.abs(librosa.stft(y))
                spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))

                # Store features and labels
                features.append([
                    np.mean(spectral_centroid),
                    np.mean(zcr),
                    np.mean(spectral_flux)
                ])
                labels.append(label)
                file_names.append(audio_file)

            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    return np.array(features), labels, file_names

def normalize_features(features):
    """
    Normalize features using StandardScaler.
    """
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

# Define folder paths
audio_folder = ["Samples_Drumset", "Chords"]  # Replace with the actual paths

# Extract features
features, labels, file_names = extract_features(audio_folder)

# Normalize features
if features.size > 0:
    normalized_features = normalize_features(features)

    # Create a DataFrame
    df = pd.DataFrame(normalized_features, columns=['Spectral Centroid', 'Zero-Crossing Rate', 'Spectral Flux'])
    df['Label'] = labels
    df['File Name'] = file_names

    # Save to CSV
    output_path = "Audio_Features_Classification.csv"
    df.to_csv(output_path, sep=';', index=False)
    print(f"Features saved to {output_path}")
