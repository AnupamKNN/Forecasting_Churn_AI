import os, sys

from src.forecast_churn.exception.exception import ForecastChurnException
from src.forecast_churn.logging.logger import logging

from src.forecast_churn.entity.config_entity import ModelTrainerConfig
from src.forecast_churn.entity.artifact_entity import (DataTransformationArtifact, ModelTrainerArtifact, 
                                                       RegressionModelTrainerArtifact, ClassificationModelTrainerArtifact, TrainingArtifactsBundle)

from src.forecast_churn.utils.main_utils.utils import save_object, load_object
from src.forecast_churn.utils.main_utils.utils import load_numpy_array_data
from src.forecast_churn.utils.main_utils.utils import evaluate_regression_model, evaluate_classification_model
from src.forecast_churn.utils.ml_utils.model.classification_estimator import ClassificationModel
from src.forecast_churn.utils.ml_utils.model.regression_estimator import RegressionModel
from src.forecast_churn.utils.ml_utils.metric.classification_metric import get_classification_score    
from src.forecast_churn.utils.ml_utils.metric.regression_metric import get_regression_score


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from xgboost import XGBClassifier, XGBRegressor

import mlflow

from dotenv import load_dotenv
load_dotenv()


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ForecastChurnException(e, sys)

    def track_regression_mlflow(self, best_model, train_metrics, test_metrics):
        with mlflow.start_run(run_name="Regression Model"):
            # Train Metrics
            mlflow.log_metric("train_r2_score", train_metrics.r2_score)
            mlflow.log_metric("train_mae", train_metrics.mean_absolute_error)
            mlflow.log_metric("train_mse", train_metrics.mean_squared_error)
            mlflow.log_metric("train_rmse", train_metrics.rmse)

            # Test Metrics
            mlflow.log_metric("test_r2_score", test_metrics.r2_score)
            mlflow.log_metric("test_mae", test_metrics.mean_absolute_error)
            mlflow.log_metric("test_mse", test_metrics.mean_squared_error)
            mlflow.log_metric("test_rmse", test_metrics.rmse)

            # Model
            mlflow.sklearn.log_model(best_model, "regression_model")


    def track_classification_mlflow(self, best_model, train_metrics, test_metrics):
        with mlflow.start_run(run_name="Classification Model"):
            # Train Metrics
            mlflow.log_metric("train_f1_score", train_metrics.f1_score)
            mlflow.log_metric("train_precision", train_metrics.precision_score)
            mlflow.log_metric("train_recall", train_metrics.recall_score)
            mlflow.log_metric("train_accuracy", train_metrics.accuracy_score)

            # Test Metrics
            mlflow.log_metric("test_f1_score", test_metrics.f1_score)
            mlflow.log_metric("test_precision", test_metrics.precision_score)
            mlflow.log_metric("test_recall", test_metrics.recall_score)
            mlflow.log_metric("test_accuracy", test_metrics.accuracy_score)

            # Model
            mlflow.sklearn.log_model(best_model, "classification_model")

        
    def train_regression_model(self, x_train, y_train, x_test, y_test)-> RegressionModel:
        try:
            models = {
            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "AdaBoostRegressor": AdaBoostRegressor(),
            "XGBRegressor": XGBRegressor(ttree_method="hist", device="cuda")
                }
            
            param_grid = {
                "LinearRegression":
                    {
                        "fit_intercept": [True, False],
                        "copy_X": [True, False],
                        "positive": [True, False]
                    },
                "GradientBoostingRegressor": {
                        "n_estimators": [100, 200, 300],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "max_depth": [3, 5, 7],
                        "subsample": [0.8, 1.0]
                    },
                "XGBRegressor": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5, 7],
                        "subsample": [0.8, 1.0],
                        "colsample_bytree": [0.8, 1.0]
                    },
                "AdaBoostRegressor": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1.0],
                        "loss": ['linear', 'square', 'exponential']
                    },
                "DecisionTreeRegressor": {
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    },
                "RandomForestRegressor": {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 2]
                    }
            }

            model_report, best_models = evaluate_regression_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, 
                                                    models = models, param = param_grid)
            
            # Extract test_r2 scores from each model
            test_r2_scores = {model_name: metrics["test_r2"] for model_name, metrics in model_report.items()}

            # Get the best model based on test r2_score
            best_model_name = max(test_r2_scores, key=test_r2_scores.get)
            best_model_score = test_r2_scores[best_model_name]
            best_model = best_models[best_model_name]


            y_train_pred = best_model.predict(x_train)
            regression_train_metrics = get_regression_score(y_train, y_train_pred)

            # Track the experiments with MLFlow
            # self.track_regression_mlflow(best_model, regression_train_metrics)

            y_test_pred = best_model.predict(x_test)
            reression_test_metrics = get_regression_score(y_test, y_test_pred)

            # Track the experiments with MLFlow
            self.track_regression_mlflow(best_model, regression_train_metrics,reression_test_metrics)

            preprocessor = load_object(file_path= self.data_transformation_artifact.transformed_regression_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.regression_model_trainer_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Regresson_Model = RegressionModel(preprocessor=preprocessor, model=best_model)
            save_object(file_path = self.model_trainer_config.regression_model_trainer_file_path, obj = Regresson_Model)
            save_object("final_model/regression_model.pkl", best_model)

            model_trainer_artifact = RegressionModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.regression_model_trainer_file_path,
                train_metric=regression_train_metrics,
                test_metric=reression_test_metrics
            )
            logging.info("Regression Model Trainer Artifact: %s", model_trainer_artifact)
            # print(best_model_score)

            return model_trainer_artifact
        
        except Exception as e:
            raise ForecastChurnException(e, sys)

    def train_classification_model(self, x_train, y_train, x_test, y_test)-> ClassificationModel:
        try:
            models = {
                "LogisticRegression": LogisticRegression(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "XGBClassifier": XGBClassifier(tree_method='hist',device='cuda')
            }
            
            param_grid = {
                "LogisticRegression":{
                        "penalty": ["l1", "l2", "elasticnet", None],
                        "C": [0.01, 0.1, 1, 10],
                        "solver": ["saga"],
                        "max_iter": [100, 200]
                    },
                "DecisionTreeClassifier": {
                        "criterion": ["gini", "entropy"],
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    },
                "RandomForestClassifier": {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 2],
                        "bootstrap": [True, False]
                    },
                "GradientBoostingClassifier": {
                        "n_estimators": [100, 200, 300],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "max_depth": [3, 5, 7],
                        "subsample": [0.8, 1.0]
                    },
                "AdaBoostClassifier": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1.0],
                        "algorithm": ['SAMME', 'SAMME.R']
                    },
                "XGBClassifier": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5, 7],
                        "subsample": [0.8, 1.0],
                        "colsample_bytree": [0.8, 1.0]
                    }
            }

            model_report, best_models = evaluate_classification_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, 
                                                        models = models, param = param_grid)
            
            # Extract test_accuracy scores from each model
            test_accuracy_scores = {model_name: metrics["test_accuracy"] for model_name, metrics in model_report.items()}

            # Get the best model based on test accuracy
            best_model_name = max(test_accuracy_scores, key=test_accuracy_scores.get)
            best_model_score = test_accuracy_scores[best_model_name]
            best_model = best_models[best_model_name]


            y_train_pred = best_model.predict(x_train)
            classification_train_metrics = get_classification_score(y_train, y_train_pred)

            # Track the experiments with MLFlow
            # self.track_classification_mlflow(best_model, classification_train_metrics)


            y_test_pred = best_model.predict(x_test)
            classification_test_metrics = get_classification_score(y_test, y_test_pred)

            # Track the experiments with MLFlow
            self.track_classification_mlflow(best_model, classification_train_metrics, classification_test_metrics)

            preprocessor = load_object(file_path= self.data_transformation_artifact.transformed_classification_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.classification_model_trainer_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Classification_Model = ClassificationModel(preprocessor=preprocessor, model=best_model)
            save_object(file_path = self.model_trainer_config.classification_model_trainer_file_path, obj = Classification_Model)
            save_object("final_model/classification_model.pkl", best_model)

            # model_trainer_artifact = ModelTrainerArtifact(
            #     trained_regression_model_file_path=self.model_trainer_config.regression_model_trainer_file_path,
            #     trained_classification_model_file_path=self.model_trainer_config.classification_model_trainer_file_path,
            #     train_regression_metric_artifact=None,
            #     test_regression_metric_artifact=None,
            #     train_classification_metric_artifact=classification_train_metrics,
            #     test_classification_metric_artifact=classification_test_metrics
            # )

            model_trainer_artifact = ClassificationModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.classification_model_trainer_file_path,
                train_metric=classification_train_metrics,
                test_metric=classification_test_metrics
            )
            logging.info(f"Classification Model trainer artifact: {model_trainer_artifact}")
            # print(best_model_score)
            return model_trainer_artifact
        except Exception as e:
            raise ForecastChurnException(e, sys)
        
    def initiate_model_trainer(self)-> ModelTrainerArtifact:
        try:
            regression_train_file_path = self.data_transformation_artifact.transformed_regression_train_file_path
            regression_test_file_path = self.data_transformation_artifact.transformed_regression_test_file_path

            classification_train_file_path = self.data_transformation_artifact.transformed_classification_train_file_path
            classification_test_file_path = self.data_transformation_artifact.transformed_classification_test_file_path

            ## Loading Train array and Test array

            regression_train_arr = load_numpy_array_data(regression_train_file_path)
            regression_test_arr = load_numpy_array_data(regression_test_file_path)

            classification_train_arr = load_numpy_array_data(classification_train_file_path)
            classification_test_arr = load_numpy_array_data(classification_test_file_path)

            x_train = regression_train_arr[:10000, :-1]
            y_train = regression_train_arr[:10000, -1]

            x_test = regression_test_arr[:10000, :-1]
            y_test = regression_test_arr[:10000, -1]

            x_train_classification = classification_train_arr[:10000, :-1]
            y_train_classification = classification_train_arr[:10000, -1]

            x_test_classification = classification_test_arr[:10000, :-1]
            y_test_classification = classification_test_arr[:10000, -1]

            regression_artifact = self.train_regression_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

            classification_artifact = self.train_classification_model(x_train=x_train_classification, y_train=y_train_classification,
                                                                      x_test=x_test_classification, y_test=y_test_classification)

            return TrainingArtifactsBundle(
            regression_artifact=regression_artifact,
            classification_artifact=classification_artifact
            )

            

            # return self.train_regression_model(x_train=x_train, 
            #                                    y_train=y_train, 
            #                                    x_test=x_test, 
            #                                    y_test=y_test), self.train_classification_model(x_train=x_train_classification, 
            #                                                                                    y_train=y_train_classification, 
            #                                                                                    x_test=x_test_classification, 
            #                                                                                    y_test=y_test_classification)
        except Exception as e:
            raise ForecastChurnException(e, sys)