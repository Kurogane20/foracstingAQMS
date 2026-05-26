import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_trigger_tune_returns_202(client):
    with patch("app.routers.training.retrain_sensor") as mock_retrain:
        response = client.post("/tune/test_uid")
        assert response.status_code == 202
        assert response.json()["uid"] == "test_uid"
