import yaml
from src.forecast_churn.exception.exception import ForecastChurnException
from src.forecast_churn.logging.logger import logging
from sklearn.metrics import accuracy_score, r2_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import os, sys
import numpy as np
# import dill
import pickle

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ForecastChurnException(e, sys)
    
def write_yaml_file(file_path: str, content = object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise ForecastChurnException(e, sys)
    

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Saves numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise ForecastChurnException(e, sys)
    
def save_object(file_path: str, obj: object):
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise ForecastChurnException(e, sys)
    
def load_object(file_path: str)-> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise ForecastChurnException(e, sys)
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ForecastChurnException(e, sys)
    

def evaluate_regression_model(X_train, y_train, X_test, y_test, models, param) -> dict:
    try:
        report = {}

        # Store best estimators
        best_models = {}


        for model_name in models:
            model = models[model_name]
            para = param[model_name]

            gs = RandomizedSearchCV(model, para, cv=3, n_jobs=-1, random_state=42)
            gs.fit(X_train, y_train)

            y_train_pred = gs.predict(X_train)
            y_test_pred = gs.predict(X_test)

            report[model_name] = {
                "train_r2": r2_score(y_train, y_train_pred),
                "train_mae": mean_absolute_error(y_train, y_train_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),

                "test_r2": r2_score(y_test, y_test_pred),
                "test_mae": mean_absolute_error(y_test, y_test_pred),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            }
            best_models[model_name] = gs.best_estimator_

        return report, best_models

    except Exception as e:
        raise ForecastChurnException(e, sys)

    

def evaluate_classification_model(X_train, y_train, X_test, y_test, models, param) -> dict:
    try:
        report = {}

        # Store best estimators
        best_models = {}

        for model_name in models:
            model = models[model_name]
            para = param[model_name]

            gs = RandomizedSearchCV(model, para, cv=3, n_jobs=-1, random_state=42)
            gs.fit(X_train, y_train)

            y_train_pred = gs.predict(X_train)
            y_test_pred = gs.predict(X_test)

            report[model_name] = {
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_f1": f1_score(y_train, y_train_pred),
                "train_precision": precision_score(y_train, y_train_pred),
                "train_recall": recall_score(y_train, y_train_pred),

                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "test_f1": f1_score(y_test, y_test_pred),
                "test_precision": precision_score(y_test, y_test_pred),
                "test_recall": recall_score(y_test, y_test_pred),
            }

            best_models[model_name] = gs.best_estimator_

        return report, best_models

    except Exception as e:
        raise ForecastChurnException(e, sys)
