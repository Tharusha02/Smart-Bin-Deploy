import os
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Path to the TFLite model
MODEL_PATH = os.getenv("MODEL_PATH", "./artifacts/waste_classification_model.tflite")

# Define class labels
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

class PredictPipeline:
    def __init__(self):
        # Load the TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict_waste_type(self, image_file):
        print("in prediction pipeline")
        """
        Predicts the waste type for a given image file.

        Args:
            image_file: A file-like object containing the image data.

        Returns:
            str: The predicted waste class.
        """
        try:
            # Read and preprocess the image
            img = Image.open(io.BytesIO(image_file.read()))
            img = img.convert("RGB")  # Ensure the image is in RGB format (remove alpha channel)
            img = img.resize((224, 224))  # Resize to model input size
            img_array = np.array(img) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension and ensure float32

            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)

            # Run inference
            self.interpreter.invoke()

            # Get the output tensor
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Get the predicted class
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_class = class_labels[predicted_index]

            return predicted_class

        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            raise ValueError("Failed to process the image. Please ensure the file is a valid image.")