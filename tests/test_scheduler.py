from datetime import datetime, timezone

import pytest

from app.scheduler import BAWAAN_MENIT, detik_hingga_menit, menit_jadwal, scheduler_aktif

UTC = timezone.utc


def test_sebelum_menit_jadwal_menunggu_di_jam_yang_sama():
    assert detik_hingga_menit(datetime(2026, 11, 24, 6, 30, 0, tzinfo=UTC), 55) == 25 * 60


def test_setelah_menit_jadwal_menunggu_jam_berikutnya():
    assert detik_hingga_menit(datetime(2026, 11, 24, 6, 56, 30, tzinfo=UTC), 55) == 58 * 60 + 30


def test_tepat_di_menit_jadwal_tidak_jalan_dua_kali():
    assert detik_hingga_menit(datetime(2026, 11, 24, 6, 55, 0, tzinfo=UTC), 55) == 3600


def test_melewati_pergantian_hari():
    assert detik_hingga_menit(datetime(2026, 11, 24, 23, 59, 0, tzinfo=UTC), 55) == 56 * 60


@pytest.mark.parametrize("nilai,harap", [(None, BAWAAN_MENIT), ("10", 10), ("0", 0), ("60", BAWAAN_MENIT),
                                         ("-1", BAWAAN_MENIT), ("abc", BAWAAN_MENIT)])
def test_menit_jadwal_dari_env(monkeypatch, nilai, harap):
    if nilai is None:
        monkeypatch.delenv("PREDICT_SCHEDULE_MINUTE", raising=False)
    else:
        monkeypatch.setenv("PREDICT_SCHEDULE_MINUTE", nilai)
    assert menit_jadwal() == harap


@pytest.mark.parametrize("nilai,harap", [(None, True), ("1", True), ("0", False), ("false", False),
                                         ("OFF", False), ("no", False)])
def test_scheduler_aktif_dari_env(monkeypatch, nilai, harap):
    if nilai is None:
        monkeypatch.delenv("INTERNAL_SCHEDULER", raising=False)
    else:
        monkeypatch.setenv("INTERNAL_SCHEDULER", nilai)
    assert scheduler_aktif() is harap
