## Download dataset
```bash
import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasserh/titanic-dataset")

print("Path to dataset files:", path)
```

## DAG Design
```text
                ingest_data
                      |
               validate_data
                      |
            ---------------------
            |                   |
   handle_missing         feature_engineering
            |                   |
            ------- join -------
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

## Mount data to Airflow in docker-compose.yaml
```yaml
    volumes:
      - ${AIRFLOW_PROJ_DIR:-.}/data:/opt/airflow/data
```