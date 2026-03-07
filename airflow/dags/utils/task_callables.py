"""
This module contains the task callables for the ML pipeline.
"""
import pandas as pd 

RAW_PATH = "/opt/airflow/data/raw/titanic.csv"
PROCESSED_PATH = "/opt/airflow/data/processed/"

# Helper for Task 1
def load_dataset(**context):

    df = pd.read_csv(RAW_PATH)

    print("Dataset Shape:", df.shape)
    print("Missing Values:")
    print(df.isnull().sum())

    context['ti'].xcom_push(key="dataset_path", value=RAW_PATH)

# Helper for Task 2
def validate_data(**context):
    
    # Retrieve the dataset from Task 1
    path = context['ti'].xcom_pull(key="dataset_path", task_ids="data_ingestion")

    df = pd.read_csv(path)

    missing = df.isnull().sum()
    age = missing.Age
    embarked = missing.Embarked

    row, col = df.shape
    age_percent = (age / row) * 100
    embarked_percent = (embarked / row) * 100

    print(f"Age missing %: {age_percent:.2f}")
    print(f"Embarked missing %: {embarked_percent:.2f}")

    if age_percent > 30 or embarked_percent > 30:
        raise Exception("Missing values in Age or Embarked columns exceed 30%")

# Helper for Task 3


