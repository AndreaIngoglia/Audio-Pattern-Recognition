import os
import random
import shutil
from pathlib import Path

import pandas as pd


def extract_chord_name(filename):
	"""
	Extract the chord name (label) from the filename.
	Assumes the format: 'audio_<chord>.png'
	"""
	return '_'.join(filename.split('_')[2:4]).replace('.png', '')


def split_dataset_with_labels(input_dir, output_dir, train_ratio=0.7):
	"""
	Split dataset into training and testing sets, extracting labels.
	"""
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)
	labels = []

	for chord_dir in input_dir.iterdir():
		if chord_dir.is_dir():
			files = list(chord_dir.glob('*.png'))  # Get all spectrogram files
			random.shuffle(files)  # Shuffle files to randomize the split

			train_split = int(len(files) * train_ratio)
			train_files = files[:train_split]
			test_files = files[train_split:]

			# Create directories for train and test
			train_dir = output_dir / 'Train' / chord_dir.name
			test_dir = output_dir / 'Test' / chord_dir.name
			train_dir.mkdir(parents=True, exist_ok=True)
			test_dir.mkdir(parents=True, exist_ok=True)

			# Move files into respective directories and store labels
			for file in train_files:
				shutil.copy(file, train_dir / file.name)
				labels.append({'filename': file.name, 'label': extract_chord_name(file.name), 'split': 'train'})
			for file in test_files:
				shutil.copy(file, test_dir / file.name)
				labels.append({'filename': file.name, 'label': extract_chord_name(file.name), 'split': 'test'})

			print(f"Chord: {chord_dir.name} -> Train: {len(train_files)}, Test: {len(test_files)}")

	return labels


# Paths
input_dir = '../Chord_Spectrograms'
output_dir = '../Spectrograms_Train_Test_CNN'

# Perform the split and extract labels
labels = split_dataset_with_labels(input_dir, output_dir)
