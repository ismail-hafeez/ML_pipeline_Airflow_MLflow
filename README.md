## Download dataset
```bash
import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasserh/titanic-dataset")

print("Path to dataset files:", path)
```

## DAG Design
```text
                data_ingestion
                      |
               data_validation
                      |
            -------------------------
            |                       |
   handle_missing_values      feature_engineering
            |                       |
            ----- merge_data --------
                      |
                encode_features
                      |
                 train_model
                      |
                evaluate_model
                      |
                check_accuracy
                /            \
        register_model     reject_model
```

## Docker Compose Configuration

### MLflow Service
```yaml
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root mlflow-artifacts:/ --serve-artifacts --artifacts-destination /mlflow/artifacts
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
    restart: always
```

### MLflow Tracking URI (Environment Variable)
```yaml
    MLFLOW_TRACKING_URI: http://mlflow:5000
```

### Mount Data Volume
```yaml
    volumes:
      - ${AIRFLOW_PROJ_DIR:-.}/data:/opt/airflow/data
```

### Persistent Volumes
```yaml
volumes:
  postgres-db-volume:
  mlflow-data:
```

## How to Run
```bash
# Build and start all services (Airflow + MLflow)
docker-compose build
docker-compose up -d

# Access Airflow UI
http://localhost:8080

# Access MLflow UI (runs inside Docker, no need to start separately)
http://localhost:5000
```
