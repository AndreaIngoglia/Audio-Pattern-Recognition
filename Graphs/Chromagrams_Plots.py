import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter


def apply_cqt_threshold(CQT_mag, threshold_factor=0.1):
    """
    Apply a threshold to the CQT magnitude to reduce noise.
    :param CQT_mag: Magnitude of the CQT
    :param threshold_factor: Proportion of the max value to keep
    :return: Thresholded CQT magnitude
    """

    threshold = np.median(CQT_mag) + threshold_factor * np.std(CQT_mag)
    CQT_mag[CQT_mag < threshold] = 0
    return CQT_mag

def compute_and_plot_chromagrams(directory_path, bins_per_octave=36, n_bins=252, smoothing_window_size=12):
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    for filename in os.listdir(directory_path):
        if filename.endswith('.wav') and 'gn_j' in filename.lower():
            file_path = os.path.join(directory_path, filename)
            try:
                # Load the audio file with its original sampling rate
                y, sr = librosa.load(file_path, sr=None)

                # Compute the Constant Q-Transform (CQT)
                CQT = librosa.cqt(y, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins)
                CQT_mag = np.abs(CQT)

                # Save the unfiltered CQT for comparison
                CQT_mag_unfiltered = np.copy(CQT_mag)

                # Apply threshold to reduce noise
                CQT_mag_filtered = apply_cqt_threshold(CQT_mag)

                # Convert CQT magnitudes to dB
                ref_level = np.percentile(CQT_mag_filtered, 99)
                CQT_db_filtered = librosa.amplitude_to_db(CQT_mag_filtered, ref=ref_level)
                CQT_db_unfiltered = librosa.amplitude_to_db(CQT_mag_unfiltered,
                                                            ref=np.percentile(CQT_mag_unfiltered, 99))

                # Extract Chroma feature from the filtered CQT
                chromagram = librosa.feature.chroma_cqt(C=CQT_mag_filtered, sr=sr, bins_per_octave=bins_per_octave)
                chromagram_smoothed = median_filter(chromagram, size=(1, smoothing_window_size))

                # Plot CQT before and after filtering, and the Chromagram in a one-column layout
                plt.figure(figsize=(6, 11))

                # CQT Before Filtering
                plt.subplot(3, 1, 1)
                librosa.display.specshow(CQT_db_unfiltered, sr=sr, x_axis='time', y_axis='cqt_note',
                                         bins_per_octave=bins_per_octave, cmap='coolwarm')
                plt.title(f'Unfiltered CQT of {filename}')
                plt.colorbar(format='%+2.0f dB')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Frequency')

                # CQT After Filtering
                plt.subplot(3, 1, 2)
                librosa.display.specshow(CQT_db_filtered, sr=sr, x_axis='time', y_axis='cqt_note',
                                         bins_per_octave=bins_per_octave, cmap='coolwarm')
                plt.title(f'Filtered CQT of {filename}')
                plt.colorbar(format='%+2.0f dB')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Frequency')

                # Chromagram
                plt.subplot(3, 1, 3)
                librosa.display.specshow(chromagram_smoothed, x_axis='time', y_axis='chroma', cmap='coolwarm', sr=sr)
                plt.title(f'Chromagram of {filename}')
                plt.colorbar(label='Intensity')
                plt.yticks(range(12), pitch_classes)
                plt.xlabel('Time (seconds)')
                plt.ylabel('Pitch Class')

                # Show the plots
                plt.tight_layout()
                plt.subplots_adjust(hspace=0.6)
                plt.show()

            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Directory containing WAV files
directory_path = ['../Chords']
for path in directory_path:
    compute_and_plot_chromagrams(path)
