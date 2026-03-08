"""
This module contains the task callables for the ML pipeline.
"""
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RAW_PATH = "/opt/airflow/data/raw/Titanic-Dataset.csv"
PROCESSED_PATH = "/opt/airflow/data/processed/"
FINAL_PATH = "/opt/airflow/data/final/"
MODEL_PATH = "/opt/airflow/data/model/"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

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
def handle_missing_values(**context):
    
    path = context['ti'].xcom_pull(key="dataset_path", task_ids="data_ingestion")
    df = pd.read_csv(path)

    # Fill missing Age with median
    median_age = df['Age'].median()
    df['Age'].fillna(median_age, inplace=True)

    # Fill missing Embarked with mode
    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'].fillna(mode_embarked, inplace=True)

    # Save processed data
    df.to_csv(f"{PROCESSED_PATH}/missing_handled.csv", index=False)
    context['ti'].xcom_push(key="missing_handled_path", value=f"{PROCESSED_PATH}/missing_handled.csv")

# Helper for Task 4
def feature_engineering(**context):
    
    path = context['ti'].xcom_pull(key="dataset_path", task_ids="data_ingestion")
    df = pd.read_csv(path)

    # Create new features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int) 

    # Save processed data
    df.to_csv(f"{PROCESSED_PATH}/feature_engineering.csv", index=False)
    context['ti'].xcom_push(key="feature_engineering_path", value=f"{PROCESSED_PATH}/feature_engineering.csv")

# Helper for Task 5
def merge_data(**context):

    ti = context["ti"]

    missing_path = ti.xcom_pull(
        key="missing_handled_path",
        task_ids="missing_values"
    )

    feature_path = ti.xcom_pull(
        key="feature_engineering_path",
        task_ids="feature_engineering"
    )

    df_missing = pd.read_csv(missing_path)
    df_features = pd.read_csv(feature_path)

    # Add engineered columns to cleaned dataset
    df_missing["FamilySize"] = df_features["FamilySize"]
    df_missing["IsAlone"] = df_features["IsAlone"]

    df_missing.to_csv(f'{PROCESSED_PATH}/merged_dataset.csv', index=False)

    ti.xcom_push(
        key="merged_dataset_path",
        value=f'{PROCESSED_PATH}/merged_dataset.csv'
    )

    print("Datasets merged successfully.")

# Helper for Task 6
def encode_data(**context):
    
    path = context['ti'].xcom_pull(key="merged_dataset_path", task_ids="merge_data")
    df = pd.read_csv(path)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

    # Drop irrelevant columns
    drop_cols = ["Cabin", "Name", "Ticket", "PassengerId"]
    df.drop(columns=drop_cols, inplace=True)

    # Save processed data
    df.to_csv(f"{FINAL_PATH}/final_dataset.csv", index=False)
    context['ti'].xcom_push(key="final_dataset_path", value=f"{FINAL_PATH}/final_dataset.csv")

# Helper for Task 7
def train_model(**context):

    ti = context["ti"]

    path = ti.xcom_pull(
        key="final_dataset_path",
        task_ids="encode_data"
    )

    df = pd.read_csv(path)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameters
    model_type = "LogisticRegression"
    max_iter = 1000       # number of iterations
    C = 1.0               # regularization strength (inverse)
    penalty = "l2"        # regularization type
    solver = "liblinear"  # works well for small datasets

    model = LogisticRegression(
        max_iter=max_iter,
        C=C,
        penalty=penalty,
        solver=solver
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Titanic_Survival")

    with mlflow.start_run(run_name="LogisticRegression_5") as run:

        # Training code...
        run_id = run.info.run_id
        ti.xcom_push(key="mlflow_run_id", value=run_id)
        
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("solver", solver)

        # Train model
        model.fit(X_train, y_train)

        # Log dataset size
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_param("dataset_columns", df.shape[1])

        # Log model artifact
        mlflow.sklearn.log_model(
            model,
            artifact_path="titanic_model"
        )

        joblib.dump(model, f'{MODEL_PATH}/model.pkl')

        ti.xcom_push(key="model_path", value=f'{MODEL_PATH}/model.pkl')

        print("Model trained and saved.")

# Helper for Task 8
def evaluate_model(**context):

    ti = context["ti"]

    # Pull the saved model from train_model task
    model_path = ti.xcom_pull(
        key="model_path",
        task_ids="train_model"
    )

    model = joblib.load(model_path)

    # Pull the dataset
    data_path = ti.xcom_pull(
        key="final_dataset_path",
        task_ids="encode_data"
    )

    df = pd.read_csv(data_path)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Use the same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions) 
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Log metrics to MLflow (reuse the same run from train_model)
    run_id = ti.xcom_pull(key="mlflow_run_id", task_ids="train_model")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Titanic_Survival")
    with mlflow.start_run(run_id=run_id):

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

    # Push accuracy for branching
    ti.xcom_push(
        key="model_accuracy",
        value=float(accuracy)
    )

# Helper for Task 9
def check_accuracy(**context):
    """
    The function returns the task_id of the next task
    """

    ti = context["ti"]

    accuracy = ti.xcom_pull(
        key="model_accuracy",
        task_ids="evaluate_model"
    )

    print("Model Accuracy:", accuracy)

    if accuracy >= 0.80:
        print("Model approved for registration.")
        return "register_model"

    else:
        print("Model rejected due to low accuracy.")
        return "reject_model"

# Helper for Task 10
def register_model(**context):

    import mlflow

    ti = context["ti"]

    # Pull the model path and MLflow run_id
    model_path = ti.xcom_pull(key="model_path", task_ids="train_model")
    run_id = ti.xcom_pull(key="mlflow_run_id", task_ids="train_model")  

    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Titanic_Survival")

    # Register the model
    model_name = "Titanic_Survival_Model"

    mlflow.register_model(
        model_uri=f"runs:/{run_id}/titanic_model",  
        name=model_name
    )

    print(f"Model registered successfully as '{model_name}'")

# Helper for Task 11
def reject_model(**context):
    ti = context["ti"]
    accuracy = ti.xcom_pull(key="model_accuracy", task_ids="evaluate_model")
    print(f"Model rejected. Accuracy was {accuracy}, below threshold.")
    
    # Log rejection reason to the same MLflow run from train_model
    run_id = ti.xcom_pull(key="mlflow_run_id", task_ids="train_model")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Titanic_Survival")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("rejection_reason", f"Accuracy {accuracy:.2f} below threshold 0.80")
