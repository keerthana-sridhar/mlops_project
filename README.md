# Malaria Cell Classification MLOps Application

End-to-end MLOps system for malaria cell image classification with reproducible training, containerized deployment, experiment tracking, operational retraining, and monitoring.

## Overview

The repository combines:

- `Streamlit` for the user interface
- `FastAPI` for inference and operational APIs
- `DVC` for reproducible pipeline execution
- `MLflow` for experiment tracking and model registry
- `Airflow` for feedback-driven retraining workflows
- `Prometheus`, `Grafana`, and `Alertmanager` for monitoring and alerting

The application serves a registered `Production` model from MLflow when available and falls back to a bundled checkpoint at `models/cnn.pth` when the registry is empty or unavailable.

## Repository Structure

```text
.
├── src/                  # backend services, training, evaluation, and pipeline code
├── frontend/             # Streamlit application
├── airflow/              # Airflow DAGs and configuration
├── docker/               # Dockerfiles and runtime entrypoints
├── monitoring/           # Prometheus, Grafana, and Alertmanager configuration
├── docs/                 # architecture and supporting technical documentation
├── tests/                # automated tests
├── dvc.yaml              # DVC pipeline definition
├── dvc.lock              # pipeline lockfile
├── params.yaml           # pipeline and training parameters
└── docker-compose.yaml   # full application stack
```

## Running the Stack

### Prerequisites

- Docker Desktop
- Python 3.10+ for local, non-container execution

### Environment Configuration

Create a local environment file from the template:

```bash
cp .env.example .env
```

Generate a Fernet key for Airflow:

```bash
python3 -c "import os,base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
```

Set the generated value in `.env`:

```text
FERNET_KEY=<generated-value>
```

### Start Services

```bash
docker compose up --build
```

The stack exposes:

- Frontend: `http://localhost:8501`
- Backend API docs: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`
- Airflow: `http://localhost:8080`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001`
- Alertmanager: `http://localhost:9093`

Default credentials:

- Airflow: `airflow` / `airflow`
- Grafana: `admin` / `admin`

## Reproducibility

Reproducibility is implemented with `DVC`. The pipeline definition is stored in [`dvc.yaml`](./dvc.yaml), and the exact dependency and output state is captured in [`dvc.lock`](./dvc.lock).

The pipeline starts from a compressed dataset artifact:

```text
data/raw/raw_zipped.zip
```

The `prepare_raw_data` stage extracts and normalizes the archive into:

```text
data/raw/cell_images/
```

Supported archive layouts include:

```text
Parasitized/
Uninfected/
```

and:

```text
cell_images/
  Parasitized/
  Uninfected/
```

After extraction, the pipeline executes:

1. `prepare_raw_data`
2. `eda`
3. `preprocess`
4. `resize`
5. `train`
6. `evaluate`

This design allows full pipeline execution without requiring a DVC remote, as long as `data/raw/raw_zipped.zip` is present locally.

### Pipeline Execution

Run the full pipeline locally:

```bash
dvc repro
```

Run the same pipeline inside the containerized environment:

```bash
docker compose exec backend dvc repro
```

Useful inspection commands:

```bash
dvc dag
dvc status
dvc metrics show
dvc plots show
```

## Versioned Assets and Runtime Artifacts

Git tracks source code, configuration, documentation, pipeline metadata, and selected repository artifacts. This includes:

- code under `src/`, `frontend/`, `airflow/`, `docker/`, and `tests/`
- pipeline metadata such as `dvc.yaml`, `dvc.lock`, and `params.yaml`
- environment templates such as `.env.example`
- documentation under `docs/`
- the bundled serving checkpoint at `models/cnn.pth`

Runtime state and generated artifacts are intentionally excluded from Git. Ignored assets include:

- extracted and processed datasets under `data/raw/cell_images/` and `data/processed/`
- DVC cache and local DVC machine-specific configuration
- MLflow runtime state such as `mlruns/` and `shared/mlflow/`
- logs and transient runtime outputs
- local environment files such as `.env`

This separation keeps the repository portable and prevents runtime drift from being committed as source state.

## Dataset Artifact Distribution

The raw dataset archive is expected at `data/raw/raw_zipped.zip`. If distributed through GitHub, large-file handling is required when the archive exceeds regular GitHub file size limits. Typical distribution strategies include:

- Git LFS
- GitHub Release assets
- external artifact storage

The repository does not require a DVC remote for pipeline execution from this archive.

## Portability

The project supports both containerized and local execution.

### Docker

`Docker Compose` is the primary runtime for the full platform. It provides a consistent environment for the API layer, frontend, experiment tracking, orchestration, and monitoring stack.

### Local Execution

Local execution is supported for development and direct pipeline iteration. When Python dependencies are installed and `data/raw/raw_zipped.zip` is present, the DVC pipeline can be executed directly on the host with `dvc repro`.

## MLflow Integration

`MLflow` is used for experiment tracking, run metadata, model artifacts, and registry-backed serving.

At startup, model loading follows this resolution order:

1. `models:/MalariaClassifier/Production`
2. `models/cnn.pth`

This fallback keeps the inference service available even when the MLflow registry has no promoted model or the tracking service is temporarily unavailable.

MLflow runtime data is stored outside Git under:

```text
shared/mlflow/
```

## Airflow Integration

`Airflow` orchestrates the feedback-based operational retraining path. The retraining DAG handles downstream processing after new labelled feedback is made available, including preprocessing, fine-tuning, evaluation, and model promotion workflows.

## Key Endpoints

- `GET /health`
- `GET /ready`
- `POST /predict`
- `GET /api/metrics`
- `GET /pipeline/status`
- `GET /model/info`
- `POST /api/retraining/trigger`
- `GET /api/retraining/status`

## Testing

Run the automated test suite with:

```bash
pytest
```

## Configuration and Secrets

Runtime configuration is managed through environment variables. A template is provided in `.env.example`, while concrete values are supplied through a local `.env` file.

Secrets and machine-specific configuration are not committed. This includes:

- `.env`
- `.dvc/config.local`
- remote storage credentials
- local service secrets

This keeps versioned configuration separate from sensitive runtime state.

## Additional Documentation

- `docs/architecture.md`
- `docs/high_level_design.md`
- `docs/low_level_design.md`
- `docs/test_plan.md`
- `docs/user_manual.md`
- `docs/acceptance_criteria.md`
- `docs/reproducibility.md`
