import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.config import API_KEY


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_endpoint_returns_list(client):
    with patch("app.main.get_model_status", return_value=[]):
        response = client.get("/status")
    assert response.status_code == 200
    assert "sensors" in response.json()


def test_predict_one_calls_service(client, auth_headers):
    with patch("app.routers.prediction.predict_sensor") as mock:
        mock.return_value = {"uid": "uid_001", "status": "success", "predictions": []}
        response = client.post("/predict/uid_001", headers=auth_headers)
    assert response.status_code == 200
    mock.assert_called_once_with("uid_001")


def test_predict_all_calls_service_for_each_uid(client, auth_headers):
    with patch("app.routers.prediction.get_all_uids", return_value=["u1", "u2"]), \
         patch("app.routers.prediction.predict_sensor") as mock:
        mock.return_value = {"uid": "u1", "status": "success", "predictions": []}
        response = client.post("/predict/all", headers=auth_headers)
    assert response.status_code == 200
    assert mock.call_count == 2


def test_retrain_one_calls_service(client, auth_headers):
    with patch("app.routers.training.retrain_sensor") as mock:
        mock.return_value = {"uid": "uid_001", "status": "ready"}
        response = client.post("/retrain/uid_001", headers=auth_headers)
    assert response.status_code == 202
    mock.assert_called_once_with("uid_001")
