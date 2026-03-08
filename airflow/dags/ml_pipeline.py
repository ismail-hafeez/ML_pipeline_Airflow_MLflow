from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from utils.task_callables import (load_dataset, validate_data, handle_missing_values, 
                                feature_engineering, merge_data, encode_data, train_model, 
                                evaluate_model, check_accuracy, register_model, reject_model)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    description="An end-to-end Machine Learning pipeline where Apache Airflow orchestrates the workflow using a DAG and MLflow manages experiment tracking and model registry.",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2026, 3, 7),
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

    # Task 3 (Parallel)
    missing_values = PythonOperator(
        task_id="missing_values",
        python_callable=handle_missing_values
    )

    # Task 4 (Parallel)
    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering
    )

    # Task 5
    merge_data = PythonOperator(
        task_id="merge_data",
        python_callable=merge_data
    )

    # Task 6
    encode_data = PythonOperator(
        task_id="encode_data",
        python_callable=encode_data
    )

    # Task 7
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    # Task 8
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model
    )

    # Task 9
    branch_task = BranchPythonOperator(
        task_id="check_accuracy",
        python_callable=check_accuracy
    )

    # Task 10   
    register_model = PythonOperator(
    task_id="register_model",
    python_callable=register_model
    )

    # Task 11
    reject_model = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model
    )

    # End Task
    end = EmptyOperator(
        task_id="end"
    )

    # Task dependencies
    data_ingestion >> data_validation >> [missing_values, feature_engineering] >> merge_data \
    >> encode_data >> train_model >> evaluate_model >> branch_task

    branch_task >> register_model >> end
    branch_task >> reject_model >> end
