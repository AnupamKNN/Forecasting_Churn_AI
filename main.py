from src.forecast_churn.components.data_ingestion import DataIngestion
from src.forecast_churn.components.data_validation import DataValidation
from src.forecast_churn.components.data_transformation import DataTransformation
from src.forecast_churn.components.model_trainer import ModelTrainer
from src.forecast_churn.exception.exception import ForecastChurnException
from src.forecast_churn.logging.logger import logging
from src.forecast_churn.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.forecast_churn.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == "__main__":
    try:
        # Data Ingestion Stage
        traing_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=traing_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate the data ingestion.")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed.")
        print(data_ingestion_artifact)

        # Data Validation Stage
        data_validation_config = DataValidationConfig(training_pipeline_config=traing_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
        logging.info("Initiate the data validation.")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed.")
        print(data_validation_artifact)

        # Data Transformation Stage
        data_transformation_config = DataTransformationConfig(training_pipeline_config=traing_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
        logging.info("Initiate the data transformation.")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation completed.")
        print(data_transformation_artifact)

        # Model Training Stage
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=traing_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        logging.info("Initiate the model trainer.")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model Trainer completed.")
        print(model_trainer_artifact)

    except Exception as e:
        raise ForecastChurnException(e, sys)