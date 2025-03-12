import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings

import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained model (ensure the path is correct)
MODEL_PATH = os.getenv("MODEL_PATH", "./artifacts/waste_classification_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

class PredictPipeline:
    def __init__(self):
        pass

    def predict_waste_type(self, image_file):
        """
        Predicts the waste type for a given image file.

        Args:
            image_file: A file-like object containing the image data.

        Returns:
            str: The predicted waste class.
        """
        print("in the predict pipeline")
        try:
            # Validate the input file
            if not image_file:
                raise ValueError("No image file provided.")

            # Read and preprocess the image
            img = Image.open(io.BytesIO(image_file.read()))
            img = img.convert("RGB")  # Ensure the image is in RGB format (remove alpha channel)
            img = img.resize((224, 224))  # Resize to model input size
            img_array = np.array(img) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_class = class_labels[predicted_index]

            return predicted_class

        except Exception as e:
            # Log the error and re-raise it
            print(f"Prediction failed: {str(e)}")
            raise ValueError("Failed to process the image. Please ensure the file is a valid image.")