from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils.task_callables import load_dataset, validate_data

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    description="An end-to-end Machine Learning pipeline where Apache Airflow orchestrates the workflow using a DAG and MLflow manages experiment tracking and model registry.",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2026, 3, 1),
    catchup=False,
    tags=["ml", "pipeline"],
) as dag:

    # Task 1
    data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=load_dataset
    )

    # Task 2
    data_validation = PythonOperator(
        task_id="data_validation",
        python_callable=validate_data,
        retries=2
    )

    # Task dependencies
    data_ingestion >> data_validation
