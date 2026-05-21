import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_endpoint_returns_list(client):
    with patch("app.main.get_model_status", return_value=[]):
        response = client.get("/status")
    assert response.status_code == 200
    assert "sensors" in response.json()


def test_predict_one_calls_service(client):
    with patch("app.routers.prediction.predict_sensor") as mock:
        mock.return_value = {"uid": "uid_001", "status": "success", "predictions": []}
        response = client.post("/predict/uid_001")
    assert response.status_code == 200
    mock.assert_called_once_with("uid_001")


def test_predict_all_calls_service_for_each_uid(client):
    with patch("app.routers.prediction.get_all_uids", return_value=["u1", "u2"]), \
         patch("app.routers.prediction.predict_sensor") as mock:
        mock.return_value = {"uid": "u1", "status": "success", "predictions": []}
        response = client.post("/predict/all")
    assert response.status_code == 200
    assert mock.call_count == 2


def test_retrain_one_calls_service(client):
    with patch("app.routers.training.retrain_sensor") as mock:
        mock.return_value = {"uid": "uid_001", "status": "ready"}
        response = client.post("/retrain/uid_001")
    assert response.status_code == 200
    mock.assert_called_once_with("uid_001")
