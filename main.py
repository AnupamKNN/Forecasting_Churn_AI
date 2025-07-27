from src.forecast_churn.components.data_ingestion import DataIngestion
from src.forecast_churn.exception.exception import ForecastChurnException
from src.forecast_churn.logging.logger import logging
from src.forecast_churn.entity.config_entity import DataIngestionConfig
from src.forecast_churn.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == "__main__":
    try:
        traingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config=traingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion.")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)

    except Exception as e:
        raise ForecastChurnException(e, sys)