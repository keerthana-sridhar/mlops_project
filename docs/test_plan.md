# Test Plan

## Test Objectives

- verify the backend API endpoints
- verify end-to-end prediction behavior
- verify failure handling for invalid images
- verify model metadata retrieval
- verify reproducibility and retraining control paths

## Test Scope

### Functional Tests

- health and readiness endpoints
- valid image prediction
- invalid input rejection
- metrics endpoint availability
- model metadata endpoint availability

### Integration Tests

- frontend to backend communication
- backend to Prometheus scrape path
- backend to Alertmanager alert visibility
- backend to Airflow retraining status and trigger path

### Operational Tests

- DVC status from container
- full `dvc repro` execution inside Docker
- Airflow DAG trigger and run-state visibility

## Existing Automated Tests

Current automated tests already cover:

- `/health`
- `/ready`
- `/model/info`
- invalid file type behavior
- empty file handling
- valid prediction path

## Manual Test Cases

| ID | Scenario | Expected Result |
|---|---|---|
| TC-01 | Upload one valid malaria cell image | Prediction is returned with confidence |
| TC-02 | Upload corrupt file | File is rejected and logged in failure log |
| TC-03 | Upload OOD image | File is tagged OOD and stored for manual review |
| TC-04 | Open Grafana dashboard | Metrics are visible and updating |
| TC-05 | Trigger Airflow DAG from frontend | New DAG run appears with task states |
| TC-06 | Move reviewed images into labelled folders and rerun retraining | DAG processes data and promotes model when evaluation passes |
| TC-07 | Run `docker compose exec backend dvc repro` | Full DVC pipeline completes reproducibly |

## Acceptance Checks

- health and readiness endpoints return success
- predictions are available through the UI
- OOD and uncertain files are stored for human review
- metrics are visible in Prometheus and Grafana
- retraining can be triggered manually and observed in the UI
- the full training pipeline can be rerun inside Docker with DVC

## Automated Test Cases (src/tests/)

- test_health.py — verifies GET /health returns 200 and system is alive
- test_ready.py — verifies GET /ready returns readiness status
- test_predict_valid.py — verifies valid image returns prediction_label and confidence
- test_predict_invalid.py — verifies corrupt/invalid file returns 400
- test_predict_empty.py — verifies empty file returns 400
- test_model_info.py — verifies GET /model/info returns metadata
- test_pipeline_status.py — verifies structured DVC status parsing
- test_model_refresh.py — verifies model refresh endpoint behavior