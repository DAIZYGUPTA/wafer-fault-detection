import sys
import pandas as pd
from pymongo.mongo_client import MongoClient
from src.exception import CustomException
from src.logger import logging
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv("MONGO_DB_URL")

client = MongoClient(url)
DATABASE_NAME = "waferData"
COLLECTION_NAME = "sampleWafer"
try:
    
    df = pd.read_csv('notebooks\wafer_23012020_041211.csv')
    df.rename(columns={"Unnamed: 0": "wafer_id"}, inplace=True)

    #json_record = list(json.loads(df.T.to_json()).values())
    json_record = df.to_dict(orient="records")

    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
    logging.info("Data inserted successfully")

except Exception as e:
    raise CustomException(e, sys)
