import os
import pandas as pd

from src.forecast_churn.exception.exception import ForecastChurnException
from src.forecast_churn.logging.logger import logging
from src.forecast_churn.entity.artifact_entity import DataIngestionArtifact
import os
import sys
from typing import List

# Import Scikit-Learn train_test_split for splitting datasets
from sklearn.model_selection import train_test_split


## configuration for Data Ingestion Config
from src.forecast_churn.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ForecastChurnException(e, sys) from e

    def export_data_into_feature_store(self) -> tuple:
        try:
            # feature_store_dir = self.data_ingestion_config.feature_store_file_path
            # Ensure the feature store directory exists
            os.makedirs(self.data_ingestion_config.feature_store_file_path, exist_ok=True)

            # Load raw data
            regression_data = pd.read_csv(self.data_ingestion_config.regression_raw_data_path)
            classification_data = pd.read_csv(self.data_ingestion_config.classification_raw_data_path)

            # Save to feature store
            regression_file_path = os.path.join(self.data_ingestion_config.feature_store_file_path, "regression.csv")
            classification_file_path = os.path.join(self.data_ingestion_config.feature_store_file_path, "classification.csv")

            regression_data.to_csv(regression_file_path, index=False, header=True)
            classification_data.to_csv(classification_file_path, index=False, header=True)

            return regression_data, classification_data

        except Exception as e:
            raise ForecastChurnException(e, sys) from e

    def split_data_as_train_test(self, regression_df: pd.DataFrame, classification_df: pd.DataFrame) -> List[str]:
        try:
            # Perform train-test split
            regression_train, regression_test = train_test_split(
                regression_df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            classification_train, classification_test = train_test_split(
                classification_df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            logging.info("Performed train-test split on the datasets.")

            logging.info("Exited split_data_as_train_test method of Data Ingestion class.")

            # Create directories
            os.makedirs(os.path.dirname(self.data_ingestion_config.regression_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.classification_train_file_path), exist_ok=True)

            # Save files
            regression_train.to_csv(self.data_ingestion_config.regression_train_file_path, index=False, header=True)
            regression_test.to_csv(self.data_ingestion_config.regression_test_file_path, index=False, header=True)

            classification_train.to_csv(self.data_ingestion_config.classification_train_file_path, index=False, header=True)
            classification_test.to_csv(self.data_ingestion_config.classification_test_file_path, index=False, header=True)

            logging.info(f"Exported train and test file path for both regression and classification.")

            return [
                self.data_ingestion_config.regression_train_file_path,
                self.data_ingestion_config.regression_test_file_path,
                self.data_ingestion_config.classification_train_file_path,
                self.data_ingestion_config.classification_test_file_path,
            ]

        except Exception as e:
            raise ForecastChurnException(e, sys) from e

    def initiate_data_ingestion(self) -> List[str]:
        try:
            regression_df, classification_df = self.export_data_into_feature_store()
            self.split_data_as_train_test(regression_df, classification_df)
            dataingestionartifact = DataIngestionArtifact(
                regression_train_file_path=self.data_ingestion_config.regression_train_file_path,
                regression_test_file_path=self.data_ingestion_config.regression_test_file_path,
                classification_train_file_path=self.data_ingestion_config.classification_train_file_path,
                classification_test_file_path=self.data_ingestion_config.classification_test_file_path
            )
            return dataingestionartifact

        except Exception as e:
            raise ForecastChurnException(e, sys) from e