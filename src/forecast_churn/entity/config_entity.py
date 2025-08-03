from datetime import datetime
from src.forecast_churn.constants import training_pipeline
import os

print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACTS_DIR)


class TrainingPipelineConfig:
    def __init__(self, timestamp = datetime.now()):
        # timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifacts_name = training_pipeline.ARTIFACTS_DIR
        self.artifacts_dir = os.path.join(self.artifacts_name)
        # self.timestamp: str = timestamp


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

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifacts_dir, training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_regression_train_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.REGRESSION_TRAIN_FILE_NAME)
        self.valid_regression_test_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.REGRESSION_TEST_FILE_NAME)
        self.valid_classification_train_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.CLASSIFICATION_TRAIN_FILE_NAME)
        self.valid_classification_test_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.CLASSIFICATION_TEST_FILE_NAME)
        self.invalid_regression_train_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.REGRESSION_TRAIN_FILE_NAME)
        self.invalid_regression_test_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.REGRESSION_TEST_FILE_NAME)
        self.invalid_classification_train_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.CLASSIFICATION_TRAIN_FILE_NAME)
        self.invalid_classification_test_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.CLASSIFICATION_TEST_FILE_NAME)
        self.regression_drift_report_file_path: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
                                                        training_pipeline.DATA_VALIDATION_REGRESSION_DRIFT_REPORT_FILE_NAME)
        self.classification_drift_report_file_path: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
                                                        training_pipeline.DATA_VALIDATION_CLASSIFICATION_DRIFT_REPORT_FILE_NAME)
        

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifacts_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_regression_train_file_path: str = os.path.join(self.data_transformation_dir,training_pipeline.REGRESSION_TRAIN_FILE_NAME.replace("csv", "npy"))
        self.transformed_regression_test_file_path: str = os.path.join(self.data_transformation_dir,training_pipeline.REGRESSION_TEST_FILE_NAME.replace("csv", "npy"))
        self.transformed_classification_train_file_path: str = os.path.join(self.data_transformation_dir, training_pipeline.CLASSIFICATION_TRAIN_FILE_NAME.replace("csv", "npy"))
        self.transformed_classification_test_file_path: str = os.path.join(self.data_transformation_dir,training_pipeline.CLASSIFICATION_TEST_FILE_NAME.replace("csv", "npy"))
        self.transformed_regression_object_file_path: str = os.path.join(self.data_transformation_dir, training_pipeline.REGRESSION_PREPROCESSING_OBJECT_FILE_NAME)
        self.transformed_classification_object_file_path: str = os.path.join(self.data_transformation_dir, training_pipeline.CLASSIFICATION_PREPROCESSING_OBJECT_FILE_NAME)


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_traier_dir: str = os.path.join(training_pipeline_config.artifacts_dir, training_pipeline.MODEL_TRAINER_DIR_NAME)
        self.regression_model_trainer_file_path: str = os.path.join(self.model_traier_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR_REGRESSION, 
                                                                    training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME_REGRESSION)
        self.classification_model_trainer_file_path: str = os.path.join(self.model_traier_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR_CLASSIFICATION,
                                                                    training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME_CLASSIFICATION)
        self.regression_r2_score: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE_REGRESSION
        self.classification_accuracy_score: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE_CLASSIFICATION
        self.overfitting_underfitting_threshold: float = training_pipeline.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD
