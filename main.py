from src.forecast_churn.components.data_ingestion import DataIngestion
from src.forecast_churn.components.data_validation import DataValidation
from src.forecast_churn.components.data_transformation import DataTransformation
from src.forecast_churn.exception.exception import ForecastChurnException
from src.forecast_churn.logging.logger import logging
from src.forecast_churn.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from src.forecast_churn.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == "__main__":
    try:
        # Data Ingestion Stage
        traingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config=traingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion.")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed.")
        print(dataingestionartifact)

        # Data Validation Stage
        datavalidationconfig = DataValidationConfig(training_pipeline_config=traingpipelineconfig)
        data_validation = DataValidation(data_ingestion_artifact=dataingestionartifact, data_validation_config=datavalidationconfig)
        logging.info("Initiate the data validation.")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed.")
        print(data_validation_artifact)

        # Data Transformation Stage
        datatransformationconfig = DataTransformationConfig(training_pipeline_config=traingpipelineconfig)
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=datatransformationconfig)
        logging.info("Initiate the data transformation.")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation completed.")
        print(data_transformation_artifact)

    except Exception as e:
        raise ForecastChurnException(e, sys)