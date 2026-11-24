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


def test_trigger_tune_returns_202(client, auth_headers):
    with patch("app.routers.training.retrain_sensor") as mock_retrain:
        response = client.post("/tune/test_uid", headers=auth_headers)
        assert response.status_code == 202
        assert response.json()["uid"] == "test_uid"


def test_training_report_returns_saved_backtest(client, auth_headers):
    report = {"uid": "AQI01B", "targets": {"tsp": {"overall": {"mae": 30.5}}}}
    with patch("app.routers.training.load_report", return_value=report) as mock_load:
        response = client.get("/training/report/AQI01B", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == report
    mock_load.assert_called_once_with("AQI01B")


def test_training_report_404_when_not_trained(client, auth_headers):
    with patch("app.routers.training.load_report", return_value=None):
        response = client.get("/training/report/AQI99X", headers=auth_headers)
    assert response.status_code == 404


def test_training_report_rejects_path_escape(client, auth_headers):
    """uid dipakai sebagai nama folder di saved_models/."""
    with patch("app.routers.training.load_report") as mock_load:
        response = client.get("/training/report/..", headers=auth_headers)
    assert response.status_code == 404
    mock_load.assert_not_called()
