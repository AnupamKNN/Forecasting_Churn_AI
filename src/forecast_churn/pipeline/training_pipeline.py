import os, sys

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.forecast_churn.exception.exception import ForecastChurnException
from src.forecast_churn.logging.logger import logging

from src.forecast_churn.components.data_ingestion import DataIngestion
from src.forecast_churn.components.data_validation import DataValidation
from src.forecast_churn.components.data_transformation import DataTransformation
from src.forecast_churn.components.model_trainer import ModelTrainer
from src.forecast_churn.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, TrainingPipelineConfig
from src.forecast_churn.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise ForecastChurnException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Validation")
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data Validation completed and artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise ForecastChurnException(e, sys)
        

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Transformation")
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation completed and artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ForecastChurnException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Model Trainer")
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model Trainer completed and artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ForecastChurnException(e, sys)
        
    def run_pipeline(self):
        try:
            logging.info("Start Data Validation")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")
            logging.info("Start Data Validation")
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            logging.info(f"Data Validation completed and artifact: {data_validation_artifact}")
            logging.info("Start Data Transformation")
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            logging.info(f"Data Transformation completed and artifact: {data_transformation_artifact}")
            logging.info("Start Model Trainer")
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logging.info(f"Model Trainer completed and artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ForecastChurnException(e, sys)