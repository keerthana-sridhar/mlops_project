#!/bin/sh
set -eu

cd /opt/project

python src/migrate_mlflow_store.py

mkdir -p /opt/project/shared/mlflow/artifacts

exec python -m mlflow server \
  --backend-store-uri "sqlite:////opt/project/shared/mlflow/mlflow.db" \
  --default-artifact-root "mlflow-artifacts:/" \
  --artifacts-destination "/opt/project/shared/mlflow/artifacts" \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*"