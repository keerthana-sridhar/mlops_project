from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


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