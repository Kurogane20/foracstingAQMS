import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

FEATURE_COLS = [
    "pm_25", "pm_25_correction", "pm_10", "pm_10_correction",
    "tsp", "tsp_correction", "noise", "temp", "mmhg", "humidity",
    "aqi_index_pm25", "aqi_index_pm10", "aqi_index_tsp", "aqi_index",
]

N_FEATURES = len(FEATURE_COLS)   # 14
N_INPUT_HOURS = 24
N_FORECAST_HOURS = 6
SSA_WINDOW = 12
BILSTM_UNITS = 64
TRAIN_HISTORY_HOURS = 24 * 90    # 90 days

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
