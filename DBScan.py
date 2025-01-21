import os
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def extract_features(audio_folder):
    """
    Extracts Spectral Centroid, Zero-Crossing Rate, and Spectral Flux for each audio file.
    """
    features = []
    file_names = []

    for folder in audio_folder:
        audio_files = random.sample([f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3'))], 500)

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

                # Store features
                features.append([np.mean(spectral_centroid), np.mean(zcr), np.mean(spectral_flux)])
                file_names.append(audio_file)

            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    return np.array(features), file_names

def normalize_features(features):
    """
    Normalize features using StandardScaler.
    """
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

def cluster_with_dbscan(features, file_names, eps=0.2, min_samples=3):
    """
    Perform DBSCAN clustering and visualize results in 3D.
    """
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features)

    # Plot the clusters in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=cluster_labels, cmap='viridis', s=50, picker=True)
    ax.set_title("Clustering of Audio Features with DBSCAN")
    ax.set_xlabel("Spectral Centroid")
    ax.set_ylabel("Zero-Crossing Rate")
    ax.set_zlabel("Spectral Flux")
    fig.colorbar(scatter, ax=ax, label='Cluster')

    # Add interactivity for hover
    annotation = ax.text(0, 0, 0, "", fontsize=9, color='red', visible=False)

    def on_hover(event):
        if event.inaxes == ax:
            # Get the closest data point
            cont, ind = scatter.contains(event)
            if cont:
                idx = ind["ind"][0]
                x, y, z = features[idx]
                annotation.set_position((event.x, event.y))
                annotation.set_text(file_names[idx])
                annotation.set_x(x)
                annotation.set_y(y)
                annotation.set_z(z)
                annotation.set_visible(True)
                fig.canvas.draw_idle()
            else:
                annotation.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    plt.show()


audio_folder = ["Samples_Drumset", "Chords"]
features, file_names = extract_features(audio_folder)
normalized_features = normalize_features(features)
cluster_with_dbscan(normalized_features, file_names, eps=0.2, min_samples=3)