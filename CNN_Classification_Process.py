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
num_classes = len(train_generator.class_indices)  # Number of chord classes
model = create_cnn(input_shape, num_classes)

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Metric to monitor (e.g., 'val_loss' or 'val_accuracy')
    patience=5,              # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore model weights from the best epoch
)

# Train the model (20 epochs)
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,  # Increased to 20 epochs
    callbacks=[early_stopping]  # Add EarlyStopping callback
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
plt.xticks(range(1, 21))  # Adjust x-axis for 20 epochs
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
plt.xticks(range(1, 21))  # Adjust x-axis for 20 epochs
plt.grid(True)
plt.show()



"""Epoch 1/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 209s 220ms/step - accuracy: 0.0347 - loss: 3.7891 - val_accuracy: 0.2922 - val_loss: 2.4617
Epoch 2/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 213s 226ms/step - accuracy: 0.2366 - loss: 2.6195 - val_accuracy: 0.5099 - val_loss: 1.3241
Epoch 3/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 202s 214ms/step - accuracy: 0.4484 - loss: 1.7098 - val_accuracy: 0.6650 - val_loss: 0.9041
Epoch 4/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 207s 219ms/step - accuracy: 0.5508 - loss: 1.3385 - val_accuracy: 0.7344 - val_loss: 0.7356
Epoch 5/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 205s 217ms/step - accuracy: 0.6181 - loss: 1.1167 - val_accuracy: 0.7402 - val_loss: 0.6982
Epoch 6/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 214s 226ms/step - accuracy: 0.6579 - loss: 0.9792 - val_accuracy: 0.8132 - val_loss: 0.5251
Epoch 7/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 204s 216ms/step - accuracy: 0.6943 - loss: 0.8683 - val_accuracy: 0.8166 - val_loss: 0.5070
Epoch 8/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 220s 233ms/step - accuracy: 0.7234 - loss: 0.8136 - val_accuracy: 0.8180 - val_loss: 0.5178
Epoch 9/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 231s 244ms/step - accuracy: 0.7381 - loss: 0.7690 - val_accuracy: 0.8569 - val_loss: 0.3847
Epoch 10/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 218s 231ms/step - accuracy: 0.7542 - loss: 0.7018 - val_accuracy: 0.8193 - val_loss: 0.4583
Epoch 11/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 214s 226ms/step - accuracy: 0.7720 - loss: 0.6651 - val_accuracy: 0.8090 - val_loss: 0.5236
Epoch 12/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 203s 214ms/step - accuracy: 0.7871 - loss: 0.6145 - val_accuracy: 0.8836 - val_loss: 0.3028
Epoch 13/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 200s 212ms/step - accuracy: 0.8038 - loss: 0.5806 - val_accuracy: 0.8198 - val_loss: 0.4773
Epoch 14/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 196s 207ms/step - accuracy: 0.8087 - loss: 0.5596 - val_accuracy: 0.8850 - val_loss: 0.3028
Epoch 15/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 212s 225ms/step - accuracy: 0.8157 - loss: 0.5472 - val_accuracy: 0.7890 - val_loss: 0.5943
Epoch 16/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 216s 228ms/step - accuracy: 0.8220 - loss: 0.5269 - val_accuracy: 0.8709 - val_loss: 0.3514
Epoch 17/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 210s 222ms/step - accuracy: 0.8306 - loss: 0.4940 - val_accuracy: 0.8801 - val_loss: 0.3223
Epoch 18/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 181s 191ms/step - accuracy: 0.8333 - loss: 0.4880 - val_accuracy: 0.8940 - val_loss: 0.2785
Epoch 19/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 180s 191ms/step - accuracy: 0.8459 - loss: 0.4562 - val_accuracy: 0.8231 - val_loss: 0.5259
Epoch 20/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 178s 188ms/step - accuracy: 0.8509 - loss: 0.4307 - val_accuracy: 0.8509 - val_loss: 0.3903
405/405 ━━━━━━━━━━━━━━━━━━━━ 24s 59ms/step - accuracy: 0.8951 - loss: 0.2775
Test Loss: 0.2785, Test Accuracy: 0.8940"""


"V2"
"""Epoch 1/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 167s 175ms/step - accuracy: 0.0314 - loss: 3.8060 - val_accuracy: 0.1292 - val_loss: 3.0966
Epoch 2/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 194s 205ms/step - accuracy: 0.1245 - loss: 3.1473 - val_accuracy: 0.3933 - val_loss: 1.8827
Epoch 3/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 192s 203ms/step - accuracy: 0.2811 - loss: 2.3318 - val_accuracy: 0.5631 - val_loss: 1.4191
Epoch 4/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 172s 182ms/step - accuracy: 0.4052 - loss: 1.8578 - val_accuracy: 0.6927 - val_loss: 0.9308
Epoch 5/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 170s 180ms/step - accuracy: 0.4794 - loss: 1.5693 - val_accuracy: 0.6794 - val_loss: 0.8353
Epoch 6/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 171s 181ms/step - accuracy: 0.5376 - loss: 1.3695 - val_accuracy: 0.7865 - val_loss: 0.6249
Epoch 7/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 171s 181ms/step - accuracy: 0.5803 - loss: 1.2299 - val_accuracy: 0.8080 - val_loss: 0.5669
Epoch 8/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 174s 184ms/step - accuracy: 0.6154 - loss: 1.1130 - val_accuracy: 0.7986 - val_loss: 0.5329
Epoch 9/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 172s 182ms/step - accuracy: 0.6423 - loss: 1.0375 - val_accuracy: 0.8081 - val_loss: 0.5115
Epoch 10/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 172s 182ms/step - accuracy: 0.6736 - loss: 0.9422 - val_accuracy: 0.8142 - val_loss: 0.4945
Epoch 11/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 172s 182ms/step - accuracy: 0.6869 - loss: 0.9069 - val_accuracy: 0.8424 - val_loss: 0.4344
Epoch 12/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 179s 190ms/step - accuracy: 0.7064 - loss: 0.8463 - val_accuracy: 0.8114 - val_loss: 0.4698
Epoch 13/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 201s 213ms/step - accuracy: 0.7198 - loss: 0.8089 - val_accuracy: 0.8265 - val_loss: 0.4602
Epoch 14/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 213s 226ms/step - accuracy: 0.7400 - loss: 0.7587 - val_accuracy: 0.8466 - val_loss: 0.4400
Epoch 15/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 200s 212ms/step - accuracy: 0.7503 - loss: 0.7289 - val_accuracy: 0.8606 - val_loss: 0.3648
Epoch 16/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 182s 192ms/step - accuracy: 0.7587 - loss: 0.7007 - val_accuracy: 0.8110 - val_loss: 0.5036
Epoch 17/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 186s 197ms/step - accuracy: 0.7673 - loss: 0.6779 - val_accuracy: 0.8627 - val_loss: 0.3774
Epoch 18/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 204s 216ms/step - accuracy: 0.7828 - loss: 0.6385 - val_accuracy: 0.8539 - val_loss: 0.3527
Epoch 19/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 179s 189ms/step - accuracy: 0.7847 - loss: 0.6309 - val_accuracy: 0.8627 - val_loss: 0.3613
Epoch 20/20
945/945 ━━━━━━━━━━━━━━━━━━━━ 180s 191ms/step - accuracy: 0.8012 - loss: 0.5770 - val_accuracy: 0.8932 - val_loss: 0.2657
405/405 ━━━━━━━━━━━━━━━━━━━━ 24s 59ms/step - accuracy: 0.8969 - loss: 0.2593
Test Loss: 0.2657, Test Accuracy: 0.8932"""