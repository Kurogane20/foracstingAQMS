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
    from app.config import API_KEY
    if not API_KEY:
        pytest.fail("API_KEY not set — cannot test auth-protected endpoints")
    return {"X-API-Key": API_KEY}


def test_get_accuracy_uid_returns_200(client, auth_headers):
    with patch("app.routers.accuracy.resolve_prediction_actuals", return_value=3), \
         patch("app.routers.accuracy.compute_accuracy", return_value={"overall_mae": 4.5}):
        response = client.get("/accuracy/sensor_01", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["uid"] == "sensor_01"
    assert data["resolved"] == 3
    assert data["accuracy"]["overall_mae"] == 4.5


def test_get_accuracy_all_returns_200(client, auth_headers):
    with patch("app.routers.accuracy.get_all_uids", return_value=["s1", "s2"]), \
         patch("app.routers.accuracy.resolve_prediction_actuals", return_value=0), \
         patch("app.routers.accuracy.compute_accuracy", return_value={"overall_mae": 3.1}):
        response = client.get("/accuracy/all", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "s1" in data
    assert "s2" in data
