import os
import sys
from flask import Flask, render_template, request, send_file
from src.exception import CustomException
from src.logger import logging as lg

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return "Welcome to my ML application 🚀"


# ------------------ TRAIN ROUTE ------------------
@app.route("/train")
def train_route():
    try:
        lg.info("Training pipeline started")

        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()

        lg.info("Training pipeline completed")

        return "✅ Training Completed Successfully!"

    except Exception as e:
        raise CustomException(e, sys)


# ------------------ PREDICT ROUTE ------------------
@app.route('/predict', methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':

            # 1️⃣ Save uploaded file
            file = request.files['file']

            input_dir = "prediction_artifacts"
            os.makedirs(input_dir, exist_ok=True)

            input_file_path = os.path.join(input_dir, file.filename)
            file.save(input_file_path)

            lg.info(f"File saved at {input_file_path}")

            # 2️⃣ Run prediction pipeline
            prediction_pipeline = PredictionPipeline()
            output_file_path = prediction_pipeline.run_pipeline(input_file_path)

            lg.info("Prediction completed. Sending file...")

            # 3️⃣ Return prediction file
            return send_file(
                output_file_path,
                download_name=os.path.basename(output_file_path),
                as_attachment=True
            )

        else:
            return render_template('upload_file.html')

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

