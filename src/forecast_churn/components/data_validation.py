from src.forecast_churn.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.forecast_churn.entity.config_entity import DataValidationConfig
from src.forecast_churn.constants.training_pipeline import SCHEMA_FILE_PATH
from src.forecast_churn.utils.main_utils.utils import read_yaml_file, write_yaml_file
from src.forecast_churn.exception.exception import ForecastChurnException
from src.forecast_churn.logging.logger import logging
from scipy.stats import ks_2samp
import pandas as pd
import os, sys


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise ForecastChurnException(e, sys) from e
        
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ForecastChurnException(e, sys)
        
        
    def validate_regression_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = self._schema_config["REGRESSION_DATA"]["COLUMNS"]
            expected_num_columns = len(expected_columns)
            logging.info(f"[Regression] Expected columns: {expected_num_columns}")
            actual_num_columns = dataframe.shape[1]
            logging.info(f"[Regression] Actual columns: {actual_num_columns}")
            if actual_num_columns == expected_num_columns:
                return True
            else:
                return False
        except Exception as e:
            raise ForecastChurnException(e, sys)
        
    def validate_classification_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = self._schema_config["CLASSIFICATION_DATA"]["COLUMNS"]
            expected_num_columns = len(expected_columns)
            logging.info(f"[Classification] Expected columns: {expected_num_columns}")
            actual_num_columns = dataframe.shape[1]
            logging.info(f"[Classification] Actual columns: {actual_num_columns}")
            if actual_num_columns == expected_num_columns:
                return True
            else:
                return False
        except Exception as e:
            raise ForecastChurnException(e, sys)


    def is_numeric_column(self, dataframe: pd.DataFrame, column_name: str)->bool:
        try:
            return dataframe[column_name].dtypes in ["int64", "float64"]
        except Exception as e:
            raise ForecastChurnException(e, sys)
        

    def detect_dataset_drift(self, base_df, current_df, drift_report_file_path: str, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_sample_dist = ks_2samp(d1, d2)
                if threshold <= is_sample_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False

                report[column] = {
                    "p_value": float(is_sample_dist.pvalue),
                    "drift_status": is_found
                }
            drift_report_file_path = os.path.join(drift_report_file_path)

            # Create directory if it doesn't exist
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)

        except Exception as e:
            raise ForecastChurnException(e, sys)

    
        

    def initiate_data_validation(self)-> DataValidationArtifact:
        try:
            regression_train_file_path = self.data_ingestion_artifact.regression_train_file_path
            regression_test_file_path = self.data_ingestion_artifact.regression_test_file_path
            classification_train_file_path = self.data_ingestion_artifact.classification_train_file_path
            classification_test_file_path = self.data_ingestion_artifact.classification_test_file_path


            ## read the data from train and test
            regression_train_df = DataValidation.read_data(regression_train_file_path)
            regression_test_df = DataValidation.read_data(regression_test_file_path)

            classification_train_df = DataValidation.read_data(classification_train_file_path)
            classification_test_df = DataValidation.read_data(classification_test_file_path)

            # Validate number of columns
            regression_validation_status_train = self.validate_regression_number_of_columns(dataframe=regression_train_df)
            classification_validation_status_train = self.validate_classification_number_of_columns(dataframe=classification_train_df)
            regression_validation_status_test = self.validate_regression_number_of_columns(dataframe=regression_test_df)
            classification_validation_status_test = self.validate_classification_number_of_columns(dataframe=classification_test_df)

            error_message_reg = ""
            error_message_class = ""

            if not regression_validation_status_train:
                error_message_reg = f"{error_message_reg} Regression dataframe does not contain all columns.\n"

            if not classification_validation_status_train:
                error_message_class = f"{error_message_class} Classification dataframe does not contain all columns.\n"

            if not regression_validation_status_test:
                error_message_reg = f"{error_message_reg} Regression dataframe does not contain all columns.\n"

            if not classification_validation_status_test:
                error_message_class = f"{error_message_class} Classification dataframe does not contain all columns.\n"

            
            # Check for numerical columns in the regression and classification dataframes
            error_messages = []
            if not self.is_numeric_column(dataframe=regression_train_df, column_name= regression_train_df.columns[0]):
                error_messages.append(f"Regression Train Dataframe does not contain all the numeric columns")

            if not self.is_numeric_column(dataframe=classification_train_df, column_name= classification_train_df.columns[0]):
                error_messages.append(f"Classification Train Dataframe does not contain all the numeric columns")

            if not self.is_numeric_column(dataframe=regression_test_df, column_name=regression_test_df.columns[0]):
                error_messages.append(f"Regression Test Dataframe does not contain all the numeric columns")

            if not self.is_numeric_column(dataframe=classification_test_df, column_name=classification_test_df.columns[0]):
                error_messages.append(f"Classification Test Dataframe does not contain all the numeric columns")

            ## let us check datadrift for both regression and classification dataframes
            regression_status = self.detect_dataset_drift(base_df= regression_train_df, current_df=regression_test_df, 
                                                          drift_report_file_path=self.data_validation_config.regression_drift_report_file_path)
            classification_status = self.detect_dataset_drift(base_df= classification_train_df, current_df=classification_test_df
                                                          , drift_report_file_path=self.data_validation_config.classification_drift_report_file_path)

            regression_dir_path = os.path.dirname(self.data_validation_config.valid_regression_train_file_path)
            classification_dir_path = os.path.dirname(self.data_validation_config.valid_classification_train_file_path)

            os.makedirs(regression_dir_path, exist_ok=True)
            os.makedirs(classification_dir_path, exist_ok=True)

            regression_train_df.to_csv(
                path_or_buf=self.data_validation_config.valid_regression_train_file_path,
                index=False,
                header=True
            )

            classification_train_df.to_csv(
                path_or_buf=self.data_validation_config.valid_classification_train_file_path,
                index=False,
                header=True
            )

            regression_test_df.to_csv(
                path_or_buf=self.data_validation_config.valid_regression_test_file_path,
                index=False,
                header=True
            )

            classification_test_df.to_csv(
                path_or_buf=self.data_validation_config.valid_classification_test_file_path,
                index=False,
                header=True
            )

            data_validation_artifact = DataValidationArtifact(
                regression_validation_status = regression_status,
                classification_validation_status = classification_status,
                valid_regression_train_file_path = self.data_validation_config.valid_regression_train_file_path,
                valid_regression_test_file_path = self.data_validation_config.valid_regression_test_file_path,
                valid_classification_train_file_path = self.data_validation_config.valid_classification_train_file_path,
                valid_classification_test_file_path = self.data_validation_config.valid_classification_test_file_path,
                regression_drift_report_file_path = self.data_validation_config.regression_drift_report_file_path,
                classification_drift_report_file_path = self.data_validation_config.classification_drift_report_file_path,
                invalid_regression_train_file_path = None,
                invalid_regression_test_file_path = None,
                invalid_classification_train_file_path = None,
                invalid_classification_test_file_path = None
            )

            return data_validation_artifact


        except Exception as e:
            raise ForecastChurnException(e, sys)
