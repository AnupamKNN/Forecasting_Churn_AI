from dataclasses import dataclass
from typing import Optional

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


@dataclass
class ClassificationMetricArtifact:
        accuracy_score: float
        f1_score: float
        precision_score: float
        recall_score: float

@dataclass
class RegressionMetricArtifact:
        mean_absolute_error: float
        mean_squared_error: float
        rmse: float
        r2_score: float

@dataclass
class ModelTrainerArtifact:
    trained_regression_model_file_path: str
    trained_classification_model_file_path: str
    train_regression_metric_artifact: RegressionMetricArtifact
    test_regression_metric_artifact: RegressionMetricArtifact
    train_classification_metric_artifact: ClassificationMetricArtifact
    test_classification_metric_artifact: ClassificationMetricArtifact


@dataclass
class RegressionModelTrainerArtifact:
    trained_model_file_path: str
    train_metric: RegressionMetricArtifact
    test_metric: RegressionMetricArtifact


@dataclass
class ClassificationModelTrainerArtifact:
    trained_model_file_path: str
    train_metric: ClassificationMetricArtifact
    test_metric: ClassificationMetricArtifact

@dataclass
class TrainingArtifactsBundle:
    regression_artifact: Optional[RegressionModelTrainerArtifact] = None
    classification_artifact: Optional[ClassificationModelTrainerArtifact] = None
