import os
import sys
import numpy as np
import pandas as pd


"""
Defining constant variables for training pipeline (separately for regression and classification)
"""

# Pipeline name
PIPELINE_NAME: str = "forecast_churn_pipeline"

# Common artifacts directory
ARTIFACTS_DIR: str = "Artifacts"

# ========================
# Regression-specific constants
# ========================
REGRESSION_FILE_NAME: str = "processed_forecast_data.csv"
REGRESSION_TARGET_COLUMN: str = "Weekly_Sales"
REGRESSION_PIPELINE_SUBDIR: str = "regression"
REGRESSION_TRAIN_FILE_NAME: str = "train_forecast.csv"
REGRESSION_TEST_FILE_NAME: str = "test_forecast.csv"

# ========================
# Classification-specific constants
# ========================
CLASSIFICATION_FILE_NAME: str = "churn_data.csv"
CLASSIFICATION_TARGET_COLUMN: str = "Churn"
CLASSIFICATION_PIPELINE_SUBDIR: str = "classification"
CLASSIFICATION_TRAIN_FILE_NAME: str = "train_churn.csv"
CLASSIFICATION_TEST_FILE_NAME: str = "test_churn.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

import os

# Generic directory to store all models
SAVED_MODEL_DIR = os.path.join("saved_models")

# Sub-directories
REGRESSION_MODEL_DIR = os.path.join(SAVED_MODEL_DIR, "regression")
CLASSIFICATION_MODEL_DIR = os.path.join(SAVED_MODEL_DIR, "classification")

# File names
REGRESSION_MODEL_FILE_NAME = "forecast_model.pkl"
CLASSIFICATION_MODEL_FILE_NAME = "churn_model.pkl"



"""
Data Ingestion related constants for regression and classification
"""

# Shared constants

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

# Regression-specific constants
REGRESSION_RAW_DATA_FILENAME: str = "preprocessed_forecast_data.csv"
REGRESSION_RAW_DATA_PATH: str = "project_data/preprocessed_forecast_data.csv"  # where forecast_data.csv is kept

# Classification-specific constants
CLASSIFICATION_RAW_DATA_FILENAME: str = "preprocessed_churn_data.csv"
CLASSIFICATION_RAW_DATA_PATH: str = "project_data/preprocessed_churn_data.csv"  # where churn_data.csv is kept


"""
Data Validation related constant start with DATA-VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_REGRESSION_DRIFT_REPORT_FILE_NAME = "regression_drift_report.yaml"
DATA_VALIDATION_CLASSIFICATION_DRIFT_REPORT_FILE_NAME = "classification_drift_report.yaml"
REGRESSION_PREPROCESSING_OBJECT_FILE_NAME = "regression_preprocessor.pkl"
CLASSIFICATION_PREPROCESSING_OBJECT_FILE_NAME = "classification_preprocessor.pkl"


"""
Data Transformation related constant start with DATA-TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
Model Trainer related constant start with MODEL_TRAINER VAR NAME
"""


MODEL_TRAINER_DIR_NAME = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR_REGRESSION = "regression_model"
MODEL_TRAINER_TRAINED_MODEL_DIR_CLASSIFICATION = "classification_model"
MODEL_TRAINER_TRAINED_MODEL_NAME_REGRESSION = "regression_model.pkl"
MODEL_TRAINER_TRAINED_MODEL_NAME_CLASSIFICATION = "classification_model.pkl"
MODEL_TRAINER_EXPECTED_SCORE_REGRESSION = 0.70  # e.g., R² score
MODEL_TRAINER_EXPECTED_SCORE_CLASSIFICATION = 0.75  # e.g., F1 score or accuracy
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD = 0.05
