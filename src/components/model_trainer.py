import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join("artifacts", "model_trainer")
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join("config", "model.yaml")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        self.models = {
            "XGBClassifier": XGBClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(),
            "RandomForestClassifier": RandomForestClassifier()
        }

    def evaluate_models(self, x_train, y_train, x_test, y_test, models):
        try:
            report = {}

            for name, model in models.items():
                logging.info(f"Training model: {name}")

                model.fit(x_train, y_train)

                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test, y_test_pred)

                logging.info(f"{name} -> Train Score: {train_score}, Test Score: {test_score}")

                report[name] = test_score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(self, best_model_object, best_model_name, X_train, y_train):
        try:
            logging.info(f"Starting hyperparameter tuning for {best_model_name}")

            model_param_grid = self.utils.read_yaml_file(
                self.model_trainer_config.model_config_file_path
            )["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(
                best_model_object,
                param_grid=model_param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            logging.info(f"Best params for {best_model_name}: {best_params}")

            best_model = best_model_object.set_params(**best_params)

            return best_model

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training process")

            # Split features and target
            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Evaluate models
            model_report = self.evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=self.models
            )

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            best_model = self.models[best_model_name]

            # Hyperparameter tuning
            best_model = self.finetune_best_model(
                best_model_object=best_model,
                best_model_name=best_model_name,
                X_train=x_train,
                y_train=y_train
            )

            # Final training
            best_model.fit(x_train, y_train)

            # Final evaluation
            y_pred = best_model.predict(x_test)
            final_score = accuracy_score(y_test, y_pred)

            logging.info(f"Final model accuracy: {final_score}")

            # Check threshold
            if final_score < self.model_trainer_config.expected_accuracy:
                raise Exception("No model met the expected accuracy threshold")

            # Save model
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_path),
                exist_ok=True
            )

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            logging.info(f"Model saved at {self.model_trainer_config.trained_model_path}")

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)