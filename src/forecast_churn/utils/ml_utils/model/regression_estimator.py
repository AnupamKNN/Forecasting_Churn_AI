from src.forecast_churn.constants.training_pipeline import REGRESSION_MODEL_DIR, REGRESSION_MODEL_FILE_NAME
from src.forecast_churn.exception.exception import ForecastChurnException
from src.forecast_churn.logging.logger import logging

import os, sys

class RegressionModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise ForecastChurnException(e, sys)
        
    def predict(self, X):
        try:
            X_transform = self.preprocessor.transform(X)
            return self.model.predict(X_transform)
        except Exception as e:
            raise ForecastChurnException(e, sys)