"""Skema dari scripts/init_db.py harus memuat setiap kolom yang ditulis/dibaca app/db.py.

Sebelumnya lower_bounds, upper_bounds, actual_values, drift_score, dan
last_accuracy_check_at hanya ada di database pengembangan (ditambah manual), sehingga
instalasi baru gagal menyimpan setiap prediksi.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ddl(tabel: str) -> str:
    src = open(os.path.join(ROOT, "scripts", "init_db.py")).read()
    m = re.search(rf"CREATE TABLE IF NOT EXISTS {tabel} \((.*?)\n    \)", src, re.S)
    assert m, f"CREATE TABLE {tabel} tidak ditemukan"
    return m.group(1)


def test_predictions_memuat_kolom_pita_dan_aktual():
    ddl = _ddl("predictions")
    for kolom in ("lower_bounds", "upper_bounds", "actual_values"):
        assert re.search(rf"\b{kolom}\b", ddl), kolom


def test_model_metadata_memuat_kolom_lokasi_dan_drift():
    ddl = _ddl("model_metadata")
    for kolom in ("lat", "lng", "drift_score", "last_accuracy_check_at"):
        assert re.search(rf"\b{kolom}\b", ddl), kolom


def test_migrasi_menambah_kolom_yang_sama():
    src = open(os.path.join(ROOT, "migrations", "add_bounds_actuals_drift.py")).read()
    for kolom in ("lower_bounds", "upper_bounds", "actual_values", "drift_score", "last_accuracy_check_at"):
        assert f'"{kolom}"' in src, kolom
