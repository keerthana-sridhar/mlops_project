import pytest
from fastapi.testclient import TestClient
import torch
import src.app as app_module


# -------------------------------
# GLOBAL MOCK FIXTURE
# -------------------------------
@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):

    # ---- Fake model ---- #
    class FakeModel:
        def __call__(self, x):
            # Return tensor like real model
            return torch.tensor([[2.0, 1.0]])  # 2 classes

    monkeypatch.setattr(app_module, "model", FakeModel(), raising=False)

    # Ensure model is "loaded"
    monkeypatch.setattr(app_module, "model", FakeModel(), raising=False)

    # Prevent MLflow reload
    monkeypatch.setattr(app_module, "refresh_model_if_needed", lambda *a, **k: None)

    # Avoid filesystem side effects
    monkeypatch.setattr(app_module, "log_failure", lambda *a, **k: None)

    # Avoid saving files
    monkeypatch.setattr(app_module.os, "makedirs", lambda *a, **k: None)

    # Avoid writing files
    monkeypatch.setattr(app_module, "open", lambda *a, **k: None, raising=False)


client = TestClient(app_module.app)


# -------------------------------
# BASIC TESTS
# -------------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200


# -------------------------------
# INPUT VALIDATION
# -------------------------------
def test_invalid_file_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello", "text/plain")}
    )

    # YOUR APP RETURNS 400 (not 422)
    assert response.status_code == 400


def test_empty_file():
    response = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400


# -------------------------------
# VALID PREDICTION
# -------------------------------
def test_predict_valid_image():
    with open("tests/sample.png", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", f, "image/png")}
        )

    assert response.status_code == 200
    data = response.json()

    assert "prediction_label" in data
    assert "confidence" in data


# -------------------------------
# PIPELINE STATUS
# -------------------------------
def test_pipeline_status_structured(monkeypatch):

    class CompletedProcess:
        returncode = 1
        stdout = ""
        stderr = """dvc.yaml:
\tchanged deps:
\t\tmodified:           params.yaml
"""

    monkeypatch.setattr(app_module.subprocess, "run", lambda *a, **k: CompletedProcess())

    monkeypatch.setattr(
        app_module,
        "get_model_sync_status",
        lambda force_refresh=False: {
            "serving_version": 8,
            "latest_production_version": 8,
            "stale": False,
            "refreshed": False,
            "model_loaded": True,
            "last_refresh_at": None,
            "latest_production_run_id": "run-123",
        },
    )

    response = client.get("/pipeline/status")
    assert response.status_code == 200

    payload = response.json()
    assert payload["dvc"]["clean"] is False
    assert payload["dvc"]["entries"][0]["target"] == "dvc.yaml"


def test_log_invalid_increments_counter():
    before = app_module._global_counts["invalid"]

    response = client.post(
        "/log_invalid",
        json={"filename": "bad.png", "reason": "empty_file"},
    )

    assert response.status_code == 200
    assert app_module._global_counts["invalid"] == before + 1


# -------------------------------
# MODEL REFRESH
# -------------------------------
def test_model_refresh_endpoint(monkeypatch):

    monkeypatch.setattr(
        app_module,
        "get_model_sync_status",
        lambda force_refresh=False: {
            "serving_version": 9,
            "latest_production_version": 9,
            "stale": False,
            "refreshed": force_refresh,
            "model_loaded": True,
            "last_refresh_at": "2026-04-23T00:00:00+00:00",
            "latest_production_run_id": "run-456",
        },
    )

    response = client.post("/api/model/refresh")
    assert response.status_code == 200
    assert response.json()["refreshed"] is True
