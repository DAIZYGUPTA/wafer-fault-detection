import os
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

AWS_S3_BUCKET_NAME = "wafer-fault"
MONGO_DATABASE_NAME = "waferData"
MONGO_COLLECTION_NAME = "sampleWafer"

TARGET_COLUMN = "quality"

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder =  "artifacts"
