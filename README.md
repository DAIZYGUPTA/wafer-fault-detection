# 🚀 Wafer Fault Detection System (End-to-End ML Pipeline)

An end-to-end Machine Learning project that predicts whether a semiconductor wafer is **Good or Bad** based on sensor data.

This project demonstrates a **production-ready ML pipeline** including data ingestion, transformation, model training, and deployment using Flask.

---

## 📌 Problem Statement

In semiconductor manufacturing, wafers can be faulty due to irregular sensor readings. Detecting faulty wafers early helps reduce cost and improve quality.

---

## 🧠 Solution

Built a complete ML pipeline that:
- Collects data from MongoDB
- Processes and cleans sensor data
- Trains multiple ML models
- Selects the best model using evaluation metrics
- Serves predictions via a web application

---

## 🏗️ Project Architecture

MongoDB → Data Ingestion → Data Transformation → Model Training → Model Saving
↓
Flask API
↓
User Upload CSV
↓
Prediction Output


---

## ⚙️ Tech Stack

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- XGBoost
- MongoDB
- Flask
- Logging & Exception Handling

---

## 🔄 Pipeline Workflow

### 1. Data Ingestion
- Extracts data from MongoDB
- Converts into DataFrame
- Saves as CSV in artifacts

### 2. Data Transformation
- Handles missing values
- Converts data to numeric
- Applies scaling (RobustScaler)
- Splits into train/test sets

### 3. Model Training
- Trains multiple models:
  - Random Forest
  - Gradient Boosting
  - SVM
  - XGBoost
- Selects best model based on accuracy
- Performs hyperparameter tuning
- Saves final model

### 4. Prediction Pipeline
- Takes CSV input from user
- Applies same preprocessing
- Predicts wafer quality
- Outputs CSV with:
  - wafer_id
  - prediction (good/bad)

---

## 🌐 Flask API

### 🔹 Train Model

GET /train


### 🔹 Predict

POST /predict

Upload a CSV file and download predictions.

---

## 📂 Project Structure

src/
├── components/
│ ├── data_ingestion.py
│ ├── data_transformation.py
│ ├── model_trainer.py
│
├── pipeline/
│ ├── train_pipeline.py
│ ├── predict_pipeline.py
│
├── utils/
│
├── logger.py
├── exception.py

app.py
requirements.txt


---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py