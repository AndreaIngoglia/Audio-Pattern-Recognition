import lime.lime_image
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained CNN model
model_path = "cnn_spectrogram_chord_recognition.h5"
model = load_model(model_path)
print("Model loaded successfully.")

# Preprocess a single image
def preprocess_image(image_path):
    """
    Preprocess an image for model input.
    """
    img = load_img(image_path, target_size=(128, 128), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)

# Define a prediction function for LIME
def predict_fn(images):
    """
    Prediction function for LIME. Takes a list of images and returns class probabilities.
    """
    images = np.array(images) 
    images = images[..., 0:1]  # Convert RGB back to grayscale
    return model.predict(images)

# Initialize LIME explainer
explainer = lime.lime_image.LimeImageExplainer()

# Paths to example spectrograms (modify to match your dataset structure)
image_paths = [
    "Spectrograms_Train_Test_CNN/Test/Gn_j/piano_2_Gn_j_f_01.png"
]

for image_path in image_paths:
    # Preprocess the image
    image = preprocess_image(image_path).squeeze()  # Preprocess and remove batch dimension
    image_rgb = gray2rgb(image)  # Convert grayscale image to RGB for LIME compatibility

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        image_rgb,  # Input image in RGB format
        predict_fn,  # Prediction function
        top_labels=1,  # Number of labels to explain
        hide_color=0,  # Background color for occluded regions
        num_samples=1000  # Number of perturbed samples to generate
    )

    # Plot results
    plt.figure(figsize=(8, 12))

    # Original Spectrogram
    plt.subplot(2, 1, 1)
    plt.title(f"Original Spectrogram for {image_path.split('/')[-1]}")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    # LIME Explanation
    plt.subplot(2, 1, 2)
    plt.title("LIME Explanation")
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],  # Use the top predicted label
        positive_only=True,  # Highlight regions contributing positively to the prediction
        hide_rest=False,  # Show the entire image
        num_features=10,  # Number of superpixels to highlight
        min_weight=0.3  # Minimum weight for a superpixel to be shown
    )
    plt.imshow(mark_boundaries(temp, mask))  # Show the image with highlighted boundaries
    plt.axis("off")

    # Adjust spacing and display
    plt.tight_layout()
    plt.show()
