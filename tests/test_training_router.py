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
