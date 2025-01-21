import os
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def extract_chord_name(filename):
    """
    Extract the chord name from the filename.
    Assumes the chord name is in the filename format, e.g., "audio_C_Major.wav".
    """
    return '_'.join(filename.split('_')[2:4])

def save_spectrogram(audio_path, output_dir, img_size=(128, 128)):
    """
    Generate and save the log-mel spectrogram for a given audio file.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)

        CQT = librosa.cqt(y, sr=sr, bins_per_octave=36, n_bins=252)
        CQT_mag = np.abs(CQT)

        # Resize to desired dimensions
        fig, ax = plt.subplots()
        ax.axis('off')
        librosa.display.specshow(CQT_mag, sr=sr, fmax=8000, ax=ax, cmap='gray')

        # Save spectrogram image
        chord_name = extract_chord_name(os.path.basename(audio_path))
        output_path = os.path.join(output_dir, chord_name)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, os.path.basename(audio_path).replace('.wav', '.png'))
        fig.savefig(output_file, dpi=img_size[0] // 2, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"Saved spectrogram for {audio_path} as {output_file}")

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")


input_dir = '../Chords'
output_dir = '../Chord_Spectrograms'

for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        audio_path = os.path.join(input_dir, filename)
        save_spectrogram(audio_path, output_dir)

