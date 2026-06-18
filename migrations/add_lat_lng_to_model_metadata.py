"""One-time migration: add lat, lng columns to model_metadata."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mysql.connector
from app.config import RESULT_DB_CONFIG

def run():
    conn = mysql.connector.connect(**RESULT_DB_CONFIG)
    try:
        cursor = conn.cursor()
        for col, type_def in [("lat", "DOUBLE NULL"), ("lng", "DOUBLE NULL")]:
            try:
                cursor.execute(f"ALTER TABLE model_metadata ADD COLUMN `{col}` {type_def}")
                conn.commit()
                print(f"Added column: {col}")
            except mysql.connector.errors.DatabaseError as e:
                if "Duplicate column name" in str(e):
                    print(f"Column already exists: {col}")
                else:
                    raise
    finally:
        conn.close()

if __name__ == "__main__":
    run()
