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
        self.model = None

    def load_model(self):
        if self.model is None:
            MODEL_PATH = os.getenv("MODEL_PATH", "./artifacts/waste_classification_model.keras")
            self.model = tf.keras.models.load_model(MODEL_PATH)

    def predict_waste_type(self, image_file):
        try:
            # Lazy load the model
            if self.model is None:
                self.load_model()

            # Preprocess the image
            img = Image.open(io.BytesIO(image_file.read()))
            img = img.convert("RGB").resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = self.model.predict(img_array)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_class = class_labels[predicted_index]

            return predicted_class

        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            raise ValueError("Failed to process the image. Please ensure the file is a valid image.")