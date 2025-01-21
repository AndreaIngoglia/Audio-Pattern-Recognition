import os
import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def apply_cqt_threshold(CQT_mag, threshold_factor=0.1):
    """
    Apply a threshold to the CQT magnitude to reduce noise.
    """
    threshold = np.median(CQT_mag) + threshold_factor * np.std(CQT_mag)
    CQT_mag[CQT_mag < threshold] = 0
    return CQT_mag


def compute_binary_vectors(directory_path, bins_per_octave=36, n_bins=252, smoothing_window_size=12):
    """
    Compute binary vectors for the chromagram of the filtered CQT for each audio file.
    """
    pitch_classes = ['Cn', 'Df', 'Dn', 'Ef', 'En', 'Fn','Gf', 'Gn', 'Af', 'An', 'Bf', 'Bn']
    binary_vectors = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory_path, filename)
            try:
                # Load the audio file with its original sampling rate
                y, sr = librosa.load(file_path, sr=None)

                # Compute the Constant Q-Transform (CQT)
                CQT = librosa.cqt(y, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins)
                CQT_mag = np.abs(CQT)

                # Apply threshold to reduce noise
                CQT_mag_filtered = apply_cqt_threshold(CQT_mag)

                # Extract Chroma feature from the filtered CQT
                chromagram = librosa.feature.chroma_cqt(C=CQT_mag_filtered, sr=sr, bins_per_octave=bins_per_octave)
                chromagram_smoothed = median_filter(chromagram, size=(1, smoothing_window_size))

                # Sum of chroma values for each pitch class
                pitch_sums = np.sum(chromagram_smoothed, axis=1)

                # Identify the indices of the top 3 pitches
                top_3_indices = np.argsort(pitch_sums)[-3:]

                # Create a binary vector for the top 3 pitches
                binary_vector = [1 if i in top_3_indices else 0 for i in range(len(pitch_classes))]

                # Extract chord name (adapted to new format)
                chord_name = '_'.join(filename.split('_')[2:4])  # Adjust if needed

                # Append the result to the binary_vectors list
                binary_vectors.append({'chord_name': chord_name, **dict(zip(pitch_classes, binary_vector))})

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Convert binary vectors to a DataFrame
    return pd.DataFrame(binary_vectors)


def classify_with_hamming(binary_vectors_df, ground_truth_df):
    """
    Perform classification using Hamming distance.
    """
    pitch_columns = ['Cn', 'Df', 'Dn', 'Ef', 'En', 'Fn','Gf', 'Gn', 'Af', 'An', 'Bf', 'Bn']
    predictions = []
    actuals = []

    for _, file_row in binary_vectors_df.iterrows():
        # Extract the binary vector for the file
        file_vector = file_row[pitch_columns].values

        # Calculate Hamming distances to all ground truth vectors
        distances = ground_truth_df[pitch_columns].apply(lambda gt_vector: np.sum(file_vector != gt_vector.values), axis=1)

        # Find the chord with the minimum Hamming distance
        closest_match_idx = distances.idxmin()
        predicted_chord = ground_truth_df.loc[closest_match_idx, 'chord_name']

        # Append prediction and actual chord name
        predictions.append(predicted_chord)
        actuals.append(file_row['chord_name'])

    # Compute classification metrics
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='weighted', zero_division=1)
    recall = recall_score(actuals, predictions, average='weighted', zero_division=1)
    f1 = f1_score(actuals, predictions, average='weighted', zero_division=1)

    return accuracy, precision, recall, f1


directory_path = 'Chords'
# Compute binary vectors
binary_vectors_df = compute_binary_vectors(directory_path)

# Save to CSV
binary_vectors_df.to_csv('Binary_Vectors.csv', sep=';', index=False)
binary_vectors_df = binary_vectors_df[['chord_name', 'Cn', 'Df', 'Dn', 'Ef', 'En', 'Fn','Gf', 'Gn', 'Af', 'An', 'Bf', 'Bn']]

# Load ground truth
ground_truth_df = pd.read_csv('Ground_Truth_Full.csv', sep=';')

# Perform classification and compute metrics
accuracy, precision, recall, f1 = classify_with_hamming(binary_vectors_df, ground_truth_df)

# Print the results
classification_results = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

print(binary_vectors_df.to_string())
print(classification_results)
