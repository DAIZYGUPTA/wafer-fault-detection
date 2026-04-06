import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion()

            feature_store_file_path = data_ingestion.initiate_data_ingestion()

            logging.info(f"Data ingestion completed. File path: {feature_store_file_path}")
            return feature_store_file_path

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, feature_store_file_path):
        try:
            logging.info("Starting data transformation")

            data_transformation = DataTransformation(
                feature_store_file_path=feature_store_file_path
            )

            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()

            logging.info("Data transformation completed")
            logging.info(f"Preprocessor saved at: {preprocessor_path}")

            return train_arr, test_arr, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, train_arr, test_arr):
        try:
            logging.info("Starting model training")

            model_trainer = ModelTrainer()

            model_path = model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr
            )

            logging.info(f"Model training completed. Model saved at: {model_path}")

            return model_path

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Pipeline started")

            # Step 1: Data Ingestion
            feature_store_file_path = self.start_data_ingestion()

            # Step 2: Data Transformation
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(
                feature_store_file_path
            )

            # Step 3: Model Training
            model_path = self.start_model_training(train_arr, test_arr)

            logging.info("Pipeline completed successfully")

            print("\n Training completed successfully!")
            print(f" Model saved at: {model_path}")
            print(f" Preprocessor saved at: {preprocessor_path}")

        except Exception as e:
            raise CustomException(e, sys)


#######

"""
class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e,sys)
          
    def start_data_transformation(self,feature_store_file_path):
        try:
            data_transformation = DataTransformation(feature_store_file_path=feature_store_file_path)
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()
            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_model_training(self, train_arr,test_arr):
        try:
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(
                train_arr, test_arr
            )
            return model_score
        except Exception as e:
            raise CustomException(e,sys)
        
    def run_pipeline(self):
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_arr,test_arr,preprocessor_path = self.start_data_transformation(feature_store_file_path)
            r2_square = self.start_model_training(train_arr,test_arr)

            print(training completed. Trained model score:, r2_square)

        except Exception as e:
            raise CustomException(e,sys)

"""