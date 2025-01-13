from tensorflow.keras import layers, models


def create_cnn(input_shape, num_classes):
	model = models.Sequential([
		# First convolutional block
		layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
		layers.MaxPooling2D((2, 2)),

		# Second convolutional block
		layers.Conv2D(64, (3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),

		# Third convolutional block
		layers.Conv2D(128, (3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),

		# Flatten and fully connected layers
		layers.Flatten(),
		layers.Dense(128, activation='relu'),
		layers.Dropout(0.5),  # Regularization to prevent overfitting
		layers.Dense(num_classes, activation='softmax')  # Output layer
	])

	# Compile the model
	model.compile(optimizer='adam',
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])
	return model
