import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mysql.connector
from app.config import RESULT_DB_CONFIG


def init_db() -> None:
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id               CHAR(36) PRIMARY KEY DEFAULT (UUID()),
        uid              VARCHAR(100) NOT NULL,
        predicted_at     DATETIME NOT NULL,
        target_time      DATETIME NOT NULL,
        step             TINYINT NOT NULL,
        pm_25            DOUBLE,
        pm_25_correction DOUBLE,
        pm_10            DOUBLE,
        pm_10_correction DOUBLE,
        tsp              DOUBLE,
        tsp_correction   DOUBLE,
        noise            DOUBLE,
        temp             DOUBLE,
        mmhg             DOUBLE,
        humidity         DOUBLE,
        aqi_index_pm25   DOUBLE,
        aqi_index_pm10   DOUBLE,
        aqi_index_tsp    DOUBLE,
        aqi_index        DOUBLE,
        created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_uid_target (uid, target_time),
        INDEX idx_predicted_at (predicted_at)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_metadata (
        id                INT AUTO_INCREMENT PRIMARY KEY,
        uid               VARCHAR(100) NOT NULL UNIQUE,
        last_trained_at   DATETIME,
        last_predicted_at DATETIME,
        training_samples  INT,
        mae_score         DOUBLE,
        status            ENUM('untrained','ready','training','error') DEFAULT 'untrained',
        error_message     TEXT,
        updated_at        DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
    print("Tables created: predictions, model_metadata")


if __name__ == "__main__":
    init_db()
