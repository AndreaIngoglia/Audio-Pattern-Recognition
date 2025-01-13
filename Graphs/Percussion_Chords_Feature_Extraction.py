import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def extract_features(audio_folder):
    """
    Extracts Spectral Centroid, Zero-Crossing Rate, Spectral Flux, and Spread for each audio file.
    """
    features = []
    file_names = []

    for folder in audio_folder:
        audio_files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.flac'))]

        for audio_file in audio_files:
            audio_path = os.path.join(folder, audio_file)
            try:
                # Load the audio file
                y, sr = librosa.load(audio_path, sr=None)

                # Calculate Spectral Centroid
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

                # Calculate Spectral Spread
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

                # Calculate Zero-Crossing Rate
                zcr = librosa.feature.zero_crossing_rate(y)[0]

                # Calculate Spectral Flux
                S = np.abs(librosa.stft(y))
                spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))

                # Store features
                features.append([
                    np.mean(spectral_centroid),
                    np.mean(spectral_bandwidth),
                    np.mean(zcr),
                    np.mean(spectral_flux)
                ])
                file_names.append(audio_file)

                plot_individual_features(audio_file, sr, spectral_centroid, spectral_bandwidth, zcr, spectral_flux)

            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    return np.array(features), file_names

def plot_individual_features(audio_file, sr, spectral_centroid, spectral_bandwidth, zcr, spectral_flux):
    """
    Plot the spectral centroid, spread, zero-crossing rate, and spectral flux for an individual audio file.
    """
    plt.figure(figsize=(20, 5))

    # Plot Spectral Centroid
    plt.subplot(1, 4, 1)
    plt.plot(spectral_centroid, label='Spectral Centroid')
    plt.title(f"Spectral Centroid for {audio_file}")
    plt.xlabel("Frames")
    plt.ylabel("Hz")
    plt.legend()

    # Plot Spectral Spread
    plt.subplot(1, 4, 2)
    plt.plot(spectral_bandwidth, label='Spectral Spread')
    plt.title(f"Spectral Spread for {audio_file}")
    plt.xlabel("Frames")
    plt.ylabel("Hz")
    plt.legend()

    # Plot Zero-Crossing Rate
    plt.subplot(1, 4, 3)
    plt.plot(zcr, label='Zero-Crossing Rate')
    plt.title(f"Zero-Crossing Rate for {audio_file}")
    plt.xlabel("Frames")
    plt.ylabel("Rate")
    plt.legend()

    # Plot Spectral Flux
    plt.subplot(1, 4, 4)
    plt.plot(spectral_flux, label='Spectral Flux')
    plt.title(f"Spectral Flux for {audio_file}")
    plt.xlabel("Frames")
    plt.ylabel("Flux")
    plt.legend()

    plt.tight_layout()
    plt.show()

def normalize_features(features):
    """
    Normalize features using StandardScaler.
    """
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

# Define folder paths
audio_folder = ["../Samples_Drumset", "../Chords"]

# Extract features
features, file_names = extract_features(audio_folder)

# Normalize features
if features.size > 0:
    normalized_features = normalize_features(features)
    print("Normalized Features:")
    print(normalized_features)
