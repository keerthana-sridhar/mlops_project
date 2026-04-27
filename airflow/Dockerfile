FROM apache/airflow:3.2.0

USER airflow

# Install normal packages from PyPI
RUN pip install --no-cache-dir \
    dvc \
    matplotlib \
    mlflow \
    "numpy<2" \
    pandas \
    pillow \
    pyyaml \
    scikit-learn \
    seaborn

# Install torch from PyTorch index
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
