from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    regression_train_file_path: str
    regression_test_file_path: str
    classification_train_file_path: str
    classification_test_file_path: str


@dataclass
class DataValidationArtifact:
    regression_validation_status: bool
    classification_validation_status: bool
    valid_regression_train_file_path: str
    valid_regression_test_file_path: str
    valid_classification_train_file_path: str
    valid_classification_test_file_path: str
    invalid_regression_train_file_path: str
    invalid_regression_test_file_path: str
    invalid_classification_train_file_path: str
    invalid_classification_test_file_path: str
    regression_drift_report_file_path: str
    classification_drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_regression_train_file_path: str
    transformed_regression_test_file_path: str
    transformed_classification_train_file_path: str
    transformed_classification_test_file_path: str
    transformed_regression_object_file_path: str
    transformed_classification_object_file_path: str