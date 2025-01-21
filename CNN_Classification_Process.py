from tensorflow.keras.preprocessing.image import ImageDataGenerator
from CNN_Classificator import create_cnn
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Paths
train_dir = 'Spectrograms_Train_Test_CNN/Train'
test_dir = 'Spectrograms_Train_Test_CNN/Test'

# Data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Normalize pixel values
    width_shift_range=0.2,      # Shift horizontally up to 20% of width
    height_shift_range=0.1,     # Shift vertically up to 10% of height
    zoom_range=0.2,             # Zoom in/out by up to 20%
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training and test data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

# Create the model
input_shape = (128, 128, 1)  # Grayscale images
num_classes = len(train_generator.class_indices)  # Number of chord classes: 48
model = create_cnn(input_shape, num_classes)

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Metric to monitor
    patience=5,              # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore model weights from the best epoch
)

# Train the model (20 epochs)
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20, 
    callbacks=[early_stopping]
)

# Evaluate performance on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save('cnn_spectrogram_chord_recognition.h5')

# Plot accuracy
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(range(1, 21))  # Adjust x-axis for 20 epochs
plt.grid(True)
plt.show()

# Plot loss
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], label='Train Loss', color='blue')
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='Validation Loss', color='orange')
plt.legend()
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()

# Plot error
train_error = [1 - acc for acc in history.history['accuracy']]
val_error = [1 - acc for acc in history.history['val_accuracy']]

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_error) + 1), train_error, label='Train Error', color='blue')
plt.plot(range(1, len(val_error) + 1), val_error, label='Validation Error', color='orange')
plt.legend()
plt.title('Training - Test Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()