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
