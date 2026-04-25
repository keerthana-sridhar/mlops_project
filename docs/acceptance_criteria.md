# Acceptance Criteria

## Functional Criteria

- The frontend accepts supported image formats and displays predictions.
- The backend serves predictions from the MLflow `Production` model.
- Health and readiness endpoints are available.
- OOD and uncertain samples are captured for feedback.
- A manual trigger exists for Airflow feedback retraining.

## MLOps Criteria

- DVC can rerun the full pipeline from within Docker.
- Airflow can process newly labelled feedback and promote a model.
- MLflow tracks experiments, runs, artifacts, and model versions.
- Prometheus scrapes backend metrics.
- Grafana visualizes live monitoring metrics.
- Alertmanager forwards active alerts.

## Documentation Criteria

- Architecture diagram is available.
- High-level design is documented.
- Low-level API design is documented.
- Test plan and test cases are documented.
- User manual is available for a non-technical user.

