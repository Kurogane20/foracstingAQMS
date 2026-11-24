"""Jadwal penjadwal internal (thread di app/main.py).

Prediksi berjalan tiap jam pada MENIT TETAP (bawaan :55), bukan "tidur 3600 detik sejak
proses mulai". Dua alasan:
- Langkah 1 prakiraan adalah jam yang sedang berjalan. Pada menit 55 ember jam itu sudah
  berisi ±55 bacaan; pada menit-menit awal isinya baru segelintir.
- Dengan tidur 3600 detik, jadwal bergeser setiap kali layanan di-restart (deploy, reboot).

Variabel lingkungan:
- PREDICT_SCHEDULE_MINUTE  menit pada setiap jam (0–59, bawaan 55)
- INTERNAL_SCHEDULER       "0"/"false"/"off"/"no" mematikan thread, mis. bila prediksi
                           dipicu dari luar (penjadwal Laravel memanggil POST /predict/all)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

BAWAAN_MENIT = 55


def menit_jadwal() -> int:
    try:
        menit = int(os.getenv("PREDICT_SCHEDULE_MINUTE", BAWAAN_MENIT))
    except ValueError:
        return BAWAAN_MENIT
    return menit if 0 <= menit <= 59 else BAWAAN_MENIT


def scheduler_aktif() -> bool:
    return os.getenv("INTERNAL_SCHEDULER", "1").strip().lower() not in ("0", "false", "off", "no")


def detik_hingga_menit(sekarang: datetime, menit: int) -> float:
    """Detik dari `sekarang` sampai kemunculan berikutnya HH:menit:00 (selalu > 0)."""
    target = sekarang.replace(minute=menit, second=0, microsecond=0)
    if target <= sekarang:
        target += timedelta(hours=1)
    return (target - sekarang).total_seconds()
