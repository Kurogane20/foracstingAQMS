# tests/test_migration_accuracy.py
import mysql.connector
import pytest
from app.config import RESULT_DB_CONFIG


@pytest.fixture
def db():
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    yield conn
    conn.close()


def test_predictions_has_actual_values_column(db):
    cursor = db.cursor()
    cursor.execute("DESCRIBE predictions")
    cols = {row[0] for row in cursor.fetchall()}
    assert "actual_values" in cols


def test_model_metadata_has_drift_columns(db):
    cursor = db.cursor()
    cursor.execute("DESCRIBE model_metadata")
    cols = {row[0] for row in cursor.fetchall()}
    assert "drift_score" in cols
    assert "last_accuracy_check_at" in cols
