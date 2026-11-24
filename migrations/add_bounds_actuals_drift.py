"""Migrasi: kolom yang dipakai app/db.py tetapi belum dibuat scripts/init_db.py.

- predictions.lower_bounds / upper_bounds  : pita P10–P90 (JSON), ditulis save_predictions
- predictions.actual_values                : nilai aktual per parameter (JSON), untuk akurasi
- model_metadata.drift_score / last_accuracy_check_at : hasil cek akurasi berkala

Tanpa kolom-kolom ini setiap prediksi gagal disimpan
("Unknown column 'lower_bounds' in 'field list'"). Aman dijalankan berulang.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mysql.connector  # noqa: E402
from app.config import RESULT_DB_CONFIG  # noqa: E402

KOLOM = [
    ("predictions", "lower_bounds", "JSON NULL"),
    ("predictions", "upper_bounds", "JSON NULL"),
    ("predictions", "actual_values", "JSON NULL"),
    ("model_metadata", "drift_score", "DOUBLE NULL"),
    ("model_metadata", "last_accuracy_check_at", "DATETIME NULL"),
]


def run():
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor()
        for tabel, kolom, tipe in KOLOM:
            try:
                cursor.execute(f"ALTER TABLE `{tabel}` ADD COLUMN `{kolom}` {tipe}")
                conn.commit()
                print(f"Added column: {tabel}.{kolom}")
            except mysql.connector.errors.DatabaseError as e:
                if "Duplicate column name" in str(e):
                    print(f"Column already exists: {tabel}.{kolom}")
                else:
                    raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
