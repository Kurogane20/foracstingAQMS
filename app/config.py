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

# Cyclical time features added as extra BiLSTM inputs (not prediction targets)
TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

N_FEATURES      = len(FEATURE_COLS)   # 14
N_TIME_FEATURES = len(TIME_COLS)      # 4

METEO_COLS = [
    "meteo_wind_speed",
    "meteo_wind_dir_sin",
    "meteo_wind_dir_cos",
    "meteo_cloudcover",
    "meteo_precip",       # curah hujan jam berjalan
    "meteo_precip_3h",    # akumulasi 3 jam: menangkap permukaan yg masih basah
]
N_METEO_FEATURES = len(METEO_COLS)   # 6

# Dinaikkan setiap kali bentuk input atau transformasi berubah. Model & scaler
# tersimpan dgn versi lain tidak kompatibel dan harus dilatih ulang.
FEATURE_SCHEMA_VERSION = 4

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
# median ~50 µg/m³ tapi ekor sampai >1.000. Dengan MinMaxScaler linier, seluruh
# rentang normal tergencet ke ~3% skala dan model kehilangan resolusi di sana.
# log1p membuat sebarannya mendekati simetris, memberi model resolusi di seluruh
# rentang, sekaligus membuat galat relatif lebih berarti daripada galat absolut —
# yang memang lebih tepat untuk konsentrasi.
LOG_SCALE_COLS = [
    "pm_25", "pm_25_correction",
    "pm_10", "pm_10_correction",
    "tsp", "tsp_correction",
]
N_INPUT_HOURS   = 24
N_FORECAST_HOURS = 6
SSA_WINDOW       = 12
BILSTM_UNITS     = 128
TRAIN_HISTORY_HOURS = 24 * 365   # 1 tahun

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
