import sys, os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.forecast_churn.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.forecast_churn.entity.config_entity import DataTransformationConfig
from src.forecast_churn.exception.exception import ForecastChurnException
from src.forecast_churn.logging.logger import logging
from src.forecast_churn.constants.training_pipeline import SCHEMA_FILE_PATH, REGRESSION_TARGET_COLUMN, CLASSIFICATION_TARGET_COLUMN
from src.forecast_churn.utils.main_utils.utils import read_yaml_file, save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise ForecastChurnException(e, sys)
        
    
    @staticmethod    
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ForecastChurnException(e, sys)
        
        
    def get_data_transformer_object(self) -> Pipeline:
        """
        Initializes transformation pipelines for both regression and classification features.

        Returns:
            Tuple[Pipeline, Pipeline]: Transformation pipelines for regression and classification data.
        """
        logging.info("Entered get_data_transformer_object method of Transformation class")
        try:
            # Load schema configuration
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            # Flatten the column definitions from list of dicts to a single dict
            def extract_columns(schema_column_list):
                return {list(col.keys())[0]: list(col.values())[0] for col in schema_column_list}

            regression_columns = extract_columns(self.schema_config["REGRESSION_DATA"]["COLUMNS"])
            classification_columns = extract_columns(self.schema_config["CLASSIFICATION_DATA"]["COLUMNS"])

            # Identify numerical and categorical features
            regression_num_features = [col for col, dtype in regression_columns.items() if dtype in ["float64", "int64"]]
            regression_cat_features = [col for col, dtype in regression_columns.items() if dtype == "object"]

            classification_num_features = [col for col, dtype in classification_columns.items() if dtype in ["float64", "int64"]]
            classification_cat_features = [col for col, dtype in classification_columns.items() if dtype == "object"]

            # Pipelines for numerical and categorical features
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
            ])

            # Column transformers
            regression_processor = ColumnTransformer([
                ("num_pipeline", numeric_pipeline, regression_num_features),
                ("cat_pipeline", categorical_pipeline, regression_cat_features)
            ])

            classification_processor = ColumnTransformer([
                ("num_pipeline", numeric_pipeline, classification_num_features),
                ("cat_pipeline", categorical_pipeline, classification_cat_features)
            ])

            # Wrap in pipelines
            regression_pipeline = Pipeline([("regression_processor", regression_processor)])
            classification_pipeline = Pipeline([("classification_processor", classification_processor)])

            return regression_pipeline, classification_pipeline

        except Exception as e:
            logging.error("Error in get_data_transformer_object method")
            raise ForecastChurnException(e, sys)

        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of Data Transformation class")
        try:
            logging.info("Starting data transformation")

            # Load validated datasets
            regression_train_df = DataTransformation.read_data(self.data_validation_artifact.valid_regression_train_file_path)
            regression_test_df = DataTransformation.read_data(self.data_validation_artifact.valid_regression_test_file_path)
            classification_train_df = DataTransformation.read_data(self.data_validation_artifact.valid_classification_train_file_path)
            classification_test_df = DataTransformation.read_data(self.data_validation_artifact.valid_classification_test_file_path)

            # Extract input and target features
            def split_features(df, target_column):
                return df.drop(columns=[target_column]), df[target_column].values.reshape(-1, 1)

            regression_input_train, regression_target_train = split_features(regression_train_df, REGRESSION_TARGET_COLUMN)
            regression_input_test, regression_target_test = split_features(regression_test_df, REGRESSION_TARGET_COLUMN)
            classification_input_train, classification_target_train = split_features(classification_train_df, CLASSIFICATION_TARGET_COLUMN)
            classification_input_test, classification_target_test = split_features(classification_test_df, CLASSIFICATION_TARGET_COLUMN)

            # Load schema
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            regression_expected_columns = [list(col.keys())[0] for col in self.schema_config["REGRESSION_DATA"]["COLUMNS"]]
            classification_expected_columns = [list(col.keys())[0] for col in self.schema_config["CLASSIFICATION_DATA"]["COLUMNS"]]

            def log_column_issues(expected, actual, dataset_name):
                missing = set(expected) - set(actual)
                extra = set(actual) - set(expected)
                if missing:
                    logging.warning(f"Missing columns in {dataset_name}: {missing}")
                if extra:
                    logging.warning(f"Extra columns in {dataset_name} (will be dropped): {extra}")
                return list(set(actual).intersection(expected))  # Return valid columns only

            # Log issues and filter columns
            regression_input_train = regression_input_train[log_column_issues(regression_expected_columns, regression_input_train.columns, "regression train")]
            regression_input_test = regression_input_test[log_column_issues(regression_expected_columns, regression_input_test.columns, "regression test")]
            classification_input_train = classification_input_train[log_column_issues(classification_expected_columns, classification_input_train.columns, "classification train")]
            classification_input_test = classification_input_test[log_column_issues(classification_expected_columns, classification_input_test.columns, "classification test")]

            # Apply preprocessing
            regression_preprocessor, classification_preprocessor = self.get_data_transformer_object()

            regression_input_train_arr = regression_preprocessor.fit_transform(regression_input_train)
            regression_input_test_arr = regression_preprocessor.transform(regression_input_test)
            classification_input_train_arr = classification_preprocessor.fit_transform(classification_input_train)
            classification_input_test_arr = classification_preprocessor.transform(classification_input_test)

            # Convert sparse matrices to dense if necessary
            to_dense = lambda arr: arr.toarray() if hasattr(arr, "toarray") else arr
            regression_input_train_arr = to_dense(regression_input_train_arr)
            regression_input_test_arr = to_dense(regression_input_test_arr)
            classification_input_train_arr = to_dense(classification_input_train_arr)
            classification_input_test_arr = to_dense(classification_input_test_arr)

            # Debug prints
            logging.info(f"Regression Train Shape: {regression_input_train_arr.shape}")
            logging.info(f"Regression Test Shape: {regression_input_test_arr.shape}")
            logging.info(f"Classification Train Shape: {classification_input_train_arr.shape}")
            logging.info(f"Classification Test Shape: {classification_input_test_arr.shape}")

            # Combine features with targets
            regression_train_arr = np.c_[regression_input_train_arr, regression_target_train]
            regression_test_arr = np.c_[regression_input_test_arr, regression_target_test]
            classification_train_arr = np.c_[classification_input_train_arr, classification_target_train]
            classification_test_arr = np.c_[classification_input_test_arr, classification_target_test]

            # Save arrays and transformers
            save_numpy_array_data(self.data_transformation_config.transformed_regression_train_file_path, regression_train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_regression_test_file_path, regression_test_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_classification_train_file_path, classification_train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_classification_test_file_path, classification_test_arr)

            save_object(self.data_transformation_config.transformed_regression_object_file_path, regression_preprocessor)
            save_object(self.data_transformation_config.transformed_classification_object_file_path, classification_preprocessor)
            save_object("final_model/regression_preprocessor.pkl", regression_preprocessor)
            save_object("final_model/classification_preprocessor.pkl", classification_preprocessor)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_regression_train_file_path=self.data_transformation_config.transformed_regression_train_file_path,
                transformed_regression_test_file_path=self.data_transformation_config.transformed_regression_test_file_path,
                transformed_classification_train_file_path=self.data_transformation_config.transformed_classification_train_file_path,
                transformed_classification_test_file_path=self.data_transformation_config.transformed_classification_test_file_path,
                transformed_regression_object_file_path=self.data_transformation_config.transformed_regression_object_file_path,
                transformed_classification_object_file_path=self.data_transformation_config.transformed_classification_object_file_path,
            )

            return data_transformation_artifact


        except Exception as e:
            raise ForecastChurnException(e, sys)