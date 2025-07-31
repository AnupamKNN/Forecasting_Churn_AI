from src.forecast_churn.entity.artifact_entity import RegressionMetricArtifact
from src.forecast_churn.exception.exception import ForecastChurnException
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import sys

def get_regression_score(y_true, y_pred):
    try:
        model_mean_absolute_error = mean_absolute_error(y_true, y_pred)
        model_mean_squared_error = mean_squared_error(y_true, y_pred)
        model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        model_r2_score = r2_score(y_true, y_pred)

        regression_metric = RegressionMetricArtifact(mean_absolute_error=model_mean_absolute_error,
                                                     mean_squared_error=model_mean_squared_error,
                                                     rmse=model_rmse,
                                                     r2_score=model_r2_score)

        return regression_metric
    except Exception as e:
        raise ForecastChurnException(e, sys)