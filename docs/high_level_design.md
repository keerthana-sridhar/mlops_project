# High-Level Design

## Objective

Build an AI-assisted malaria cell classification system that supports:

- online inference for end users
- monitoring and alerting for model health
- feedback collection for uncertain and OOD samples
- retraining with manually labelled feedback
- reproducible ML pipelines

## Major Components

### 1. Frontend

- Built with Streamlit
- Used by non-technical users to upload cell images
- Displays predictions, alerts, live metrics, pipeline state, and retraining controls

### 2. Backend

- Built with FastAPI
- Exposes inference, monitoring, failure-log, pipeline-status, feedback-statistics, and retraining APIs
- Loads the current `Production` model from MLflow registry
- Saves uncertain and OOD samples into feedback queues

### 3. Model Registry and Experiment Tracking

- MLflow tracks runs, metrics, artifacts, and model versions
- Production inference always points to the `MalariaClassifier` model in `Production`

### 4. Reproducible Training Pipeline

- DVC defines the full pipeline: prepare raw data, EDA, preprocess, resize, train, evaluate
- `dvc repro` is the reproducible path for rerunning the complete tracked workflow

### 5. Feedback Retraining Pipeline

- Airflow orchestrates the shorter operational loop for newly labelled feedback
- The DAG checks for new labelled samples, processes them, fine-tunes, evaluates, and promotes the new model. It's schedule is every X mins for now.

### 6. Monitoring and Alerting

- Prometheus scrapes backend metrics
- Grafana visualizes live model and system metrics
- Alertmanager forwards active alerts back to the backend so they can be shown in the UI

## Design Rationale

- **Loose coupling**: frontend and backend communicate only over REST APIs.
- **Environment parity**: deployment uses Docker and Docker Compose.
- **Operational transparency**: monitoring, alerting, and retraining state are visible in the UI.
- **Human-in-the-loop**: OOD and uncertain predictions are reviewed by a physician before retraining.
- **Reproducibility**: full experimentation and pipeline execution are controlled by DVC and MLflow.

## Design Paradigm

The backend follows a **functional programming paradigm**:

- API logic is implemented as stateless FastAPI route handlers
- Core transformations (preprocessing, inference, metrics computation) are expressed as pure functions
- No complex class hierarchies are used for business logic

Controlled shared state exists only in:
- the loaded ML model (protected using thread-safe access)
- runtime metrics counters

This design improves:
- testability (functions can be mocked easily)
- readability
- modularity

## Logging

The backend logs:
- prediction flow
- evaluation metrics
- retraining events

Logging is used for debugging, monitoring, and traceability.
