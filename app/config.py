import os
from dotenv import load_dotenv

load_dotenv()

_required = {
    "SENSOR_DB_NAME":     os.getenv("SENSOR_DB_NAME"),
    "SENSOR_DB_USER":     os.getenv("SENSOR_DB_USER"),
    "SENSOR_DB_PASSWORD": os.getenv("SENSOR_DB_PASSWORD"),
    "RESULT_DB_NAME":     os.getenv("RESULT_DB_NAME"),
    "RESULT_DB_USER":     os.getenv("RESULT_DB_USER"),
    "RESULT_DB_PASSWORD": os.getenv("RESULT_DB_PASSWORD"),
}
_missing = [k for k, v in _required.items() if v is None]
if _missing:
    raise EnvironmentError(f"Missing required env vars: {', '.join(_missing)}")

# DB sumber data sensor (t_loggers)
SENSOR_DB_CONFIG = {
    "host":     os.getenv("SENSOR_DB_HOST", "127.0.0.1"),
    "port":     int(os.getenv("SENSOR_DB_PORT", 3306)),
    "database": _required["SENSOR_DB_NAME"],
    "user":     _required["SENSOR_DB_USER"],
    "password": _required["SENSOR_DB_PASSWORD"],
}

# DB hasil prediksi & metadata model (predictions, model_metadata)
RESULT_DB_CONFIG = {
    "host":     os.getenv("RESULT_DB_HOST", "localhost"),
    "port":     int(os.getenv("RESULT_DB_PORT", 3306)),
    "database": _required["RESULT_DB_NAME"],
    "user":     _required["RESULT_DB_USER"],
    "password": _required["RESULT_DB_PASSWORD"],
}

API_KEY = os.getenv("INTERNAL_API_KEY")

FEATURE_COLS = [
    "pm_25", "pm_25_correction", "pm_10", "pm_10_correction",
    "tsp", "tsp_correction", "noise", "temp", "mmhg", "humidity",
    "aqi_index_pm25", "aqi_index_pm10", "aqi_index_tsp", "aqi_index",
]

N_FEATURES = len(FEATURE_COLS)   # 14

# Dinaikkan setiap kali bentuk fitur atau transformasi berubah. Artefak model
# tersimpan dgn versi lain tidak kompatibel dan harus dilatih ulang.
# v5: SSA-BiLSTM-LightGBM diganti LightGBM langsung multi-horizon (lihat
# app/models/forecaster.py).
FEATURE_SCHEMA_VERSION = 5

# Batas kewajaran fisik per parameter: (min, maks, nilai_cadangan).
#
# Ini MENGGANTIKAN penyaringan statistik (3-sigma + pagar Tukey IQR) yang dipakai
# sebelumnya. Alasannya terukur: pagar Tukey per sensor jatuh di 89–252 µg/m³,
# sementara baku mutu TSP 230 µg/m³ — sehingga 7 dari 9 sensor kehilangan SELURUH
# jam pelampauannya sebelum pelatihan (59% dari 2.193 jam pelampauan terhapus).
# Model jadi tidak pernah melihat, apalagi meramalkan, kejadian yang justru paling
# penting untuk diperingatkan.
#
# Tujuan penyaringan yang benar adalah menolak SENSOR RUSAK, bukan HARI BERDEBU.
# Batas di bawah dipilih longgar: hanya nilai yang mustahil secara fisik yang
# dibuang. TSP 1.400 µg/m³ itu buruk, tapi nyata dan harus dipelajari model.
PHYSICAL_LIMITS: dict[str, tuple[float, float, float]] = {
    "pm_25":            (0.0, 2000.0, 0.0),
    "pm_25_correction": (0.0, 2000.0, 0.0),
    "pm_10":            (0.0, 3000.0, 0.0),
    "pm_10_correction": (0.0, 3000.0, 0.0),
    "tsp":              (0.0, 5000.0, 0.0),
    "tsp_correction":   (0.0, 5000.0, 0.0),
    "noise":            (20.0, 140.0, 55.0),
    # Cadangan iklim Berau: sebagian sensor tidak punya modul cuaca dan
    # melaporkan nol terus-menerus; nol akan lolos batas tapi merusak pelatihan.
    "temp":             (5.0, 55.0, 28.0),
    "mmhg":             (600.0, 820.0, 760.0),
    "humidity":         (1.0, 100.0, 82.0),
    "aqi_index_pm25":   (0.0, 500.0, 0.0),
    "aqi_index_pm10":   (0.0, 500.0, 0.0),
    "aqi_index_tsp":    (0.0, 500.0, 0.0),
    "aqi_index":        (0.0, 500.0, 0.0),
}

# Konsentrasi partikulat berdistribusi condong-kanan (mendekati lognormal):
# median ~50 µg/m³ tapi ekor sampai >1.000. Model bekerja pada log1p(nilai):
# sebarannya mendekati simetris, dan SELISIH di ruang itu setara RASIO dalam
# satuan fisik — bentuk yang tepat untuk konsentrasi. Indeks kualitas udara
# diturunkan dari konsentrasi dan mewarisi kecondongan yang sama.
LOG_SCALE_COLS = [
    "pm_25", "pm_25_correction",
    "pm_10", "pm_10_correction",
    "tsp", "tsp_correction",
    "aqi_index_pm25", "aqi_index_pm10", "aqi_index_tsp", "aqi_index",
]

N_FORECAST_HOURS = 6
TRAIN_HISTORY_HOURS = 24 * 365   # 1 tahun

# Fitur terpanjang menengok 7 hari ke belakang (jam yang sama selama sepekan);
# dua hari tambahan memberi ruang bagi celah data dan jam yang belum selesai.
PREDICT_HISTORY_HOURS = 24 * 9

# Zona waktu lokasi (WITA = UTC+8). Data disimpan UTC, tetapi siklus debu
# mengikuti jam matahari dan gilir kerja setempat, jadi fitur jam/hari dihitung
# pada jam lokal.
LOCAL_UTC_OFFSET_HOURS = int(os.getenv("LOCAL_UTC_OFFSET_HOURS", 8))

# Sensor melapor tiap menit (~60 bacaan/jam). Rata-rata dari segelintir bacaan
# bukan rata-rata jam yang bermakna; jam seperti itu diperlakukan sebagai kosong,
# baik untuk data latih maupun untuk nilai aktual pembanding.
MIN_READINGS_PER_HOUR = 10

# Jangkar prakiraan adalah nilai teramati terakhir. Bila jam terakhir kosong,
# nilai paling lama sekian jam ke belakang masih boleh dipakai; lebih tua dari
# itu, prakiraan untuk parameter tersebut tidak dibuat dari data basi.
MAX_ANCHOR_AGE_HOURS = 3

# Parameter utama untuk ringkasan skor dan tuning.
PRIMARY_PARAM = "tsp"

# ── Evaluasi rolling-origin ──────────────────────────────────────────────────
# Tiap lipatan diuji dengan model yang hanya dilatih pada data SEBELUM lipatan
# itu. Lipatan pertama hanya dipakai untuk kalibrasi (bobot penyusutan dan
# pelebaran interval); skor dilaporkan dari lipatan kedua dan seterusnya, dengan
# kalibrasi yang selalu berasal dari lipatan-lipatan sebelumnya.
BACKTEST_FOLDS = int(os.getenv("BACKTEST_FOLDS", 6))
BACKTEST_FOLD_HOURS = 24 * 28
MIN_TRAIN_HOURS = 24 * 14        # data latih minimum sebelum lipatan pertama
MIN_FOLD_HOURS = 24 * 3

# Cakupan nominal pita prakiraan (P10–P90).
INTERVAL_COVERAGE = 0.80

# Ambang untuk metrik deteksi pelampauan (POD/FAR/CSI) pada laporan backtest.
# Angkanya baku mutu udara ambien PP 22/2021 Lampiran VII — tetapi baku mutu itu
# berlaku untuk RATA-RATA 24 JAM, sedangkan prakiraan ini per jam. Metriknya
# mengukur kemampuan model memperingatkan jam berdebu tinggi; ini BUKAN
# penilaian kepatuhan terhadap baku mutu.
EXCEEDANCE_THRESHOLDS = {"tsp": 230.0, "pm_10": 75.0, "pm_25": 55.0}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
