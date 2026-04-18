FROM apache/airflow:3.2.0

USER airflow

# Install normal packages from PyPI
RUN pip install --no-cache-dir dvc mlflow

# Install torch from PyTorch index
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu