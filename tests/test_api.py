from fastapi.testclient import TestClient
import src.app as app_module

client = TestClient(app_module.app)


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


def test_invalid_file_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 422


def test_empty_file():
    response = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400


def test_predict_valid_image():
    with open("tests/sample.png", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", f, "image/png")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "prediction_label" in data


def test_pipeline_status_structured(monkeypatch):
    class CompletedProcess:
        returncode = 0
        stdout = "finetune/checkpoint.pth.dvc:\n\tchanged outs:\n\t\tmodified:           finetune/checkpoint.pth\n"
        stderr = ""

    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: CompletedProcess())
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
    assert payload["dvc"]["entries"][0]["target"] == "finetune/checkpoint.pth.dvc"
    assert payload["dvc"]["entries"][0]["category"] == "changed outs"


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
