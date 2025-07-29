from src.forecast_churn.components.data_ingestion import DataIngestion
from src.forecast_churn.components.data_validation import DataValidation
from src.forecast_churn.exception.exception import ForecastChurnException
from src.forecast_churn.logging.logger import logging
from src.forecast_churn.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.forecast_churn.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == "__main__":
    try:
        traingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config=traingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion.")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed.")
        print(dataingestionartifact)

        datavalidationconfig = DataValidationConfig(training_pipeline_config=traingpipelineconfig)
        data_validation = DataValidation(data_ingestion_artifact=dataingestionartifact, data_validation_config=datavalidationconfig)
        logging.info("Initiate the data validation.")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed.")
        print(data_validation_artifact)




    except Exception as e:
        raise ForecastChurnException(e, sys)