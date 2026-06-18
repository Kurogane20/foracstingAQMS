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
]
N_METEO_FEATURES = len(METEO_COLS)   # 4
N_INPUT_HOURS   = 24
N_FORECAST_HOURS = 6
SSA_WINDOW       = 12
BILSTM_UNITS     = 128
TRAIN_HISTORY_HOURS = 24 * 365   # 1 tahun

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
