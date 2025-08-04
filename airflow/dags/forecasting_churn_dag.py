from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# This import statement should be here, after sys.path has been modified
from src.forecast_churn.pipeline.training_pipeline import TrainingPipeline 


# Create a single pipeline instance for shared use (optional)
pipeline = TrainingPipeline()

def m_run_pipeline(**kwargs):
    pipeline.run_pipeline()

default_args = {
    "owner": "anupam",
    "retries": 0,
    "retry_delay": timedelta(minutes=1)
}

with DAG(
    dag_id="forecast_churn_training_pipeline_dag",
    default_args=default_args,
    description="Forecast Churn Modular ML training pipeline DAG",
    schedule=None,  # or use a CRON like '@daily'
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["ML", "Training", "Modular"],
) as dag:

    model_runner = PythonOperator(
        task_id="model_runner",
        python_callable=m_run_pipeline
    )

    # Define DAG dependencies
    model_runner
