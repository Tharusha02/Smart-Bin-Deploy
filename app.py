# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from src.pipeline.predict_pipeline import PredictPipeline
# from src.components.data_ingestion import DataIngestion
# from src.components.model_trainer import ModelTrainer

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
        predict_pipeline = PredictPipeline()
        # Call the prediction function
        predicted_class = predict_pipeline.predict_waste_type(file)

        # Return the result
        return jsonify({"predicted_class": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route("/train-model", methods=["POST"])
# def train_model():
#     """
#     Endpoint to train the model.
#     Expects a JSON payload with an optional 'dataset_path' key.
#     """
#     # Get dataset path from request or use default
#     data = request.get_json() or {}
#     dataset_path = data.get("dataset_path", DEFAULT_DATASET_PATH)
#
#     try:
#         # Step 1: Data Ingestion
#         print(f"Starting data ingestion for dataset path: {dataset_path}")
#         data_ingestion = DataIngestion(dataset_path)
#         train_generator, val_generator = data_ingestion.initiate_data_ingestion()
#
#         # Step 2: Model Training
#         print("Building and training the model...")
#         model_trainer = ModelTrainer(train_generator, val_generator)
#         model = model_trainer.build_model()
#         history = model_trainer.train_model(model, epochs=50)
#
#         # Step 3: Save the Model
#         model_filepath = "artifacts/waste_classification_model_new.keras"
#         model_trainer.save_model(model, filepath=model_filepath)
#
#         # Return success response
#         return jsonify(
#             {
#                 "message": "Model training completed successfully.",
#                 "model_saved_at": model_filepath,
#                 "training_history": {
#                     "epochs": len(history.history["loss"]),
#                     "final_training_accuracy": history.history["accuracy"][-1],
#                     "final_validation_accuracy": history.history["val_accuracy"][-1],
#                 },
#             }
#         )
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
#
# @app.route("/ingest-data", methods=["POST"])
# def ingest_data():
#     """
#     Endpoint to initiate data ingestion.
#     Expects a JSON payload with an optional 'dataset_path' key.
#     """
#     # Get dataset path from request or use default
#     data = request.get_json() or {}
#     dataset_path = data.get("dataset_path", DEFAULT_DATASET_PATH)
#
#     try:
#         # Initialize DataIngestion and perform ingestion
#         data_ingestion = DataIngestion(dataset_path)
#         train_generator, val_generator = data_ingestion.initiate_data_ingestion()
#
#         # Extract class names
#         class_names = list(train_generator.class_indices.keys())
#
#         # Return response
#         return jsonify(
#             {
#                 "message": "Data ingestion completed successfully.",
#                 "dataset_path": dataset_path,
#                 "class_names": class_names,
#                 "train_samples": train_generator.samples,
#                 "val_samples": val_generator.samples,
#             }
#         )
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)