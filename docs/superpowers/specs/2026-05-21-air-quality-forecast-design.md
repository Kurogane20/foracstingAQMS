# Air Quality Forecast Service — Design Spec
**Date:** 2026-05-21  
**Status:** Approved

---

## Overview

Sistem prediksi kualitas udara 6 jam ke depan berbasis SSA-BiLSTM-LightGBM, dibangun sebagai FastAPI microservice yang terintegrasi dengan sistem Laravel + MySQL yang sudah ada. Terdapat 9 sensor (uid berbeda), masing-masing memiliki model prediksi tersendiri.

---

## Context

- **Sistem existing:** Laravel dashboard + MySQL di VPS Server 138
- **Data sensor:** Per menit, 9 sensor aktif
- **Tabel sensor:** `t_loggers` dengan kolom `pm_25`, `pm_10`, `tsp`, `noise`, `temp`, `mmhg`, `humidity`, `aqi_index`, dan turunannya
- **Target prediksi:** Semua parameter raw, 6 jam ke depan (t+1 s/d t+6)
- **Model:** 9 model terpisah, satu per sensor (uid)
- **Retrain:** Otomatis setiap malam via cron job

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    VPS Server 138                    │
│                                                      │
│  ┌──────────────┐        ┌────────────────────────┐  │
│  │   Laravel    │◄──────►│   FastAPI Service      │  │
│  │  Dashboard   │ HTTP   │   (port 8001)          │  │
│  └──────┬───────┘        │  /predict/{uid}        │  │
│         │                │  /predict/all          │  │
│         │                │  /retrain/{uid}        │  │
│         │                │  /status               │  │
│         │                └──────────┬─────────────┘  │
│         ▼                           ▼                 │
│  ┌──────────────────────────────────────────────┐     │
│  │              MySQL Database                  │     │
│  │  [sensor_data — existing]                    │     │
│  │  [predictions — new]                         │     │
│  │  [model_metadata — new]                      │     │
│  └──────────────────────────────────────────────┘     │
│                                                      │
│  Cron (Laravel scheduler):                           │
│  - Setiap jam      → POST /predict/all               │
│  - Setiap 02:00    → POST /retrain/all               │
└─────────────────────────────────────────────────────┘
```

---

## Data Pipeline

```
MySQL sensor_data (per menit)
        ↓
[1. Agregasi per Jam]
  - Ambil 24 jam terakhir per uid dari tabel `t_loggers`
  - Gunakan kolom `datetime_unix` (unix timestamp dari sensor) sebagai basis waktu
  - Resample mean per jam
  - Hasil: DataFrame 24 baris × 8 kolom
        ↓
[2. Preprocessing]
  - Isi missing values: interpolasi linear
  - Deteksi outlier: IQR method (cap, tidak drop)
  - Normalisasi: MinMaxScaler per sensor (disimpan sebagai scaler_{uid}.pkl)
        ↓
[3. SSA Decomposition]
  - Window length: 12 (setengah dari 24 jam input)
  - Output: trend + oscillation per parameter
        ↓
[4. BiLSTM Multi-Output]
  - Input shape: (24 timestep, 8 parameter)
  - Output: (6 timestep, 8 parameter) — direct multi-output
  - Arsitektur: 2 layer BiLSTM (64 units) + Dropout(0.2) + Dense
        ↓
[5. LightGBM Post-processor]
  - Input: output BiLSTM + fitur waktu (jam, hari_minggu, bulan)
  - Output: prediksi final per parameter
        ↓
[6. Denormalisasi + Simpan ke tabel predictions]
```

---

## Model Files per Sensor

Disimpan di `saved_models/{uid}/`:
- `scaler.pkl` — MinMaxScaler untuk normalisasi/denormalisasi
- `bilstm.h5` — model BiLSTM terlatih
- `lgbm.pkl` — LightGBM post-processor

---

## Database Schema (Tabel Baru)

```sql
CREATE TABLE predictions (
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
);

CREATE TABLE model_metadata (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    uid               VARCHAR(100) NOT NULL UNIQUE,
    last_trained_at   DATETIME,
    last_predicted_at DATETIME,
    training_samples  INT,
    mae_score         DOUBLE,
    status            ENUM('untrained','ready','training','error') DEFAULT 'untrained',
    error_message     TEXT,
    updated_at        DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

---

## Project Structure

```
air_quality_forecast/
├── app/
│   ├── main.py                  ← FastAPI entry point
│   ├── config.py                ← DB connection, env vars
│   ├── models/
│   │   ├── preprocessor.py      ← agregasi, cleaning, normalisasi
│   │   ├── ssa.py               ← SSA decomposition
│   │   ├── bilstm.py            ← BiLSTM architecture & training
│   │   ├── lgbm.py              ← LightGBM post-processor
│   │   └── pipeline.py          ← orchestrate preprocessing → predict
│   ├── services/
│   │   ├── predict.py           ← prediction logic per sensor
│   │   └── retrain.py           ← retrain logic per sensor
│   └── routers/
│       ├── prediction.py        ← /predict endpoints
│       └── training.py          ← /retrain endpoints
├── saved_models/
│   └── {uid}/
│       ├── scaler.pkl
│       ├── bilstm.h5
│       └── lgbm.pkl
├── logs/
├── scripts/
│   └── init_db.py               ← create predictions & model_metadata tables
├── requirements.txt
└── .env
```

---

## API Endpoints

| Method | Endpoint | Fungsi |
|--------|----------|--------|
| `GET`  | `/status` | Status semua sensor: last trained, last predicted, model status |
| `POST` | `/predict/{uid}` | Prediksi 6 jam untuk satu sensor |
| `POST` | `/predict/all` | Prediksi 6 jam semua sensor |
| `POST` | `/retrain/{uid}` | Retrain model satu sensor |
| `POST` | `/retrain/all` | Retrain semua sensor |
| `GET`  | `/predictions/{uid}` | Ambil hasil prediksi terbaru satu sensor |

---

## Laravel Integration

**Service class:**
```php
// app/Services/ForecastService.php
class ForecastService {
    public function getPredictions(string $uid): array
    public function triggerPredict(string $uid): void
    public function getModelStatus(): array
}
```

**Routes baru:**
```
GET  /forecast              → dashboard prediksi semua sensor
GET  /forecast/{uid}        → detail prediksi satu sensor
POST /forecast/{uid}/predict → trigger manual prediksi
```

**Cron (Laravel scheduler):**
```php
// app/Console/Kernel.php
$schedule->call(fn() => app(ForecastService::class)->triggerPredictAll())
         ->hourly();
$schedule->call(fn() => app(ForecastService::class)->triggerRetrainAll())
         ->dailyAt('02:00');
```

---

## Dashboard Display

Setiap halaman sensor menampilkan:
1. **Chart** — Data aktual 24 jam + prediksi 6 jam ke depan dalam satu grafik, dengan dropdown pilih parameter (PM2.5, PM10, TSP, AQI, Temp, Humidity, dll.)
2. **Tabel** — 6 baris prediksi (t+1 s/d t+6) dengan semua kolom parameter

---

## Error Handling

- Sensor tidak punya cukup data historis → skip prediksi, log warning, set status `error` di `model_metadata`
- Model belum dilatih → return 404 dengan pesan jelas
- FastAPI tidak bisa dijangkau dari Laravel → Laravel tampilkan pesan "Prediksi tidak tersedia"
- Training gagal → pertahankan model lama, catat error di `model_metadata.error_message`

---

## Deployment Notes

- FastAPI dijalankan via `uvicorn` dikelola oleh **Supervisor** di VPS
- Port: 8001 (tidak expose ke publik, hanya akses internal dari Laravel)
- Virtual environment Python (`venv`) untuk isolasi dependencies
- File `.env` menyimpan kredensial DB, tidak di-commit ke git
