from datetime import datetime
from src.forecast_churn.constants import training_pipeline
import os

print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACTS_DIR)


class TrainingPipelineConfig:
    def __init__(self, timestamp = datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifacts_name = training_pipeline.ARTIFACTS_DIR
        self.artifacts_dir = os.path.join(self.artifacts_name, timestamp)
        self.timestamp: str = timestamp


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifacts_dir, training_pipeline.DATA_INGESTION_DIR_NAME   
        )

        # Paths to raw data files
        self.regression_raw_data_path: str = training_pipeline.REGRESSION_RAW_DATA_PATH
        self.classification_raw_data_path: str = training_pipeline.CLASSIFICATION_RAW_DATA_PATH

        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR
        )

        self.regression_train_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.REGRESSION_TRAIN_FILE_NAME
        )

        self.classification_train_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.CLASSIFICATION_TRAIN_FILE_NAME
        )

        self.regression_test_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.REGRESSION_TEST_FILE_NAME
        )

        self.classification_test_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.CLASSIFICATION_TEST_FILE_NAME
        )

        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION

