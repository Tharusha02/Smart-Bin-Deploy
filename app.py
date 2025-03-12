from flask import Flask, request, jsonify
from flask_cors import CORS
from src.pipeline.predict_pipeline import PredictPipeline
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Default dataset path (can be overridden via API)
DEFAULT_DATASET_PATH = "./notebook/Data/Waste_dataset"

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()

        # Call the prediction function
        predicted_class = predict_pipeline.predict_waste_type(file)

        # Return the result
        return jsonify({"predicted_class": predicted_class})

    except Exception as e:
        # Log the error and return a user-friendly message
        app.logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": "An error occurred during prediction. Please try again."}), 500

# Run the app
if __name__ == "__main__":
    # Use environment variables for configuration
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("FLASK_ENV") == "development"
    )