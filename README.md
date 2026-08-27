# Air Quality Forecast API

FastAPI service yang menyediakan prediksi kualitas udara 6 jam ke depan menggunakan model hybrid **SSA-BiLSTM-LightGBM**. Dirancang untuk diintegrasikan dengan sistem monitoring AQMS berbasis Laravel.

---

## Arsitektur Model

```
Sensor data (24h)
      │
  SSA decomposition          ← mengurangi noise, mengekstrak tren
      │
  BiLSTM                     ← menangkap pola temporal sekuensial
      │
  LightGBM                   ← residual correction + fitur tambahan
      │
  Prediksi 6 langkah (1h/step)
```

Setiap sensor (`uid`) memiliki model BiLSTM dan LightGBM tersendiri yang disimpan di `saved_models/{uid}/`.

---

## Struktur Proyek

```
air_quality_forecast/
├── app/
│   ├── auth.py              # API key dependency (require_api_key)
│   ├── config.py            # ENV vars, konstanta model, path
│   ├── db.py                # MySQL: read sensor data, simpan prediksi
│   ├── main.py              # FastAPI app entry point
│   ├── models/
│   │   ├── bilstm.py        # Build, train, predict BiLSTM (+ in-process cache)
│   │   ├── lgbm.py          # Train, predict LightGBM
│   │   ├── pipeline.py      # Pipeline train: BiLSTM → LightGBM, split data
│   │   ├── preprocessor.py  # Fetch & preprocess data sensor untuk prediksi
│   │   └── ssa.py           # Singular Spectrum Analysis decomposition
│   ├── routers/
│   │   ├── prediction.py    # POST /predict/{uid}, POST /predict/all, GET /predictions/{uid}
│   │   └── training.py      # POST /retrain/{uid}, POST /retrain/all (BackgroundTasks)
│   └── services/
│       ├── predict.py       # Orkestrasi prediksi end-to-end
│       └── retrain.py       # Orkestrasi retrain end-to-end
├── scripts/
│   └── init_db.py           # Buat tabel predictions & model_metadata
├── tests/                   # Unit tests (pytest)
├── saved_models/            # Model hasil training (di-ignore oleh git)
├── logs/                    # Log aplikasi (di-ignore oleh git)
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## Persyaratan

- Python 3.11+
- MySQL 5.7+ (database AQMS yang sudah ada, tabel `t_loggers`)

---

## Instalasi

```bash
# 1. Clone & masuk ke direktori
cd air_quality_forecast

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Salin dan isi konfigurasi
cp .env.example .env
# Edit .env sesuai konfigurasi database

# 5. Buat tabel database
python scripts/init_db.py

# 6. Jalankan server
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

---

## Konfigurasi (.env)

Service ini menggunakan **database yang sama** dengan `bc-enviro-web` (`aqms_db`). Tabel `t_loggers` dibaca dari sana, dan tabel `predictions`/`model_metadata` dibuat di database yang sama.

Nilai koneksi mengikuti `DB_AQMS_*` di `.env` Laravel:

| Variable FastAPI | Nilai | Keterangan |
|---|---|---|
| `DB_HOST` | `103.150.194.228` | Host MySQL AQMS |
| `DB_PORT` | `3306` | Port MySQL |
| `DB_NAME` | `aqms_db` | Database AQMS |
| `DB_USER` | `admin` | Username |
| `DB_PASSWORD` | *(lihat .env)* | Password |
| `INTERNAL_API_KEY` | *(sama dengan `FORECAST_API_KEY` Laravel)* | API key autentikasi |

---

## API Endpoints

### Health & Status

| Method | Endpoint | Keterangan |
|---|---|---|
| GET | `/health` | Cek service aktif |
| GET | `/status` | Status model semua sensor |

### Prediksi

| Method | Endpoint | Auth | Keterangan |
|---|---|---|---|
| POST | `/predict/{uid}` | API Key | Jalankan prediksi untuk satu sensor |
| POST | `/predict/all` | API Key | Jalankan prediksi untuk semua sensor |
| GET | `/predictions/{uid}` | — | Ambil hasil prediksi terakhir (6 baris) |

### Training

| Method | Endpoint | Auth | Keterangan |
|---|---|---|---|
| POST | `/retrain/{uid}` | API Key | Retrain model satu sensor (async, 202) |
| POST | `/retrain/all` | API Key | Retrain semua sensor (async, 202) |

**Autentikasi:** sertakan header `X-API-Key: <INTERNAL_API_KEY>` untuk endpoint yang membutuhkan auth. Jika `INTERNAL_API_KEY` kosong di `.env`, autentikasi dinonaktifkan.

---

## Format Response Prediksi

```json
{
  "uid": "SENSOR-001",
  "predictions": [
    {
      "step": 1,
      "target_time": "2024-01-01T01:00:00",
      "pm_25": 12.34,
      "pm_10": 25.67,
      "tsp": 30.12,
      "aqi_index": 45.00,
      "noise": 64.5,
      "temp": 28.3,
      "humidity": 74.1,
      "mmhg": 760.2,
      "aqi_index_pm25": 40.1,
      "aqi_index_pm10": 48.3,
      "aqi_index_tsp": 45.0
    }
    // ... step 2–6
  ]
}
```

---

## Alur Kerja

### Pertama Kali (Training)

```bash
# Jalankan prediksi — ini akan otomatis meretrain model jika belum ada
curl -X POST http://localhost:8001/predict/SENSOR-001 \
     -H "X-API-Key: your_api_key"
```

### Retrain Terjadwal

Retrain dijadwalkan dari sisi Laravel (via `routes/console.php`) setiap hari pukul 02:00. Endpoint retrain langsung merespons `202 Accepted` karena prosesnya berjalan di background.

### Prediksi On-Demand

```bash
# Prediksi satu sensor
curl -X POST http://localhost:8001/predict/SENSOR-001 \
     -H "X-API-Key: your_api_key"

# Ambil hasil terakhir (tanpa auth)
curl http://localhost:8001/predictions/SENSOR-001
```

---

## Menjalankan Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Integrasi Laravel

Service ini diintegrasikan dengan aplikasi Laravel `bc-enviro-web` melalui:

- **`App\Services\ForecastService`** — client HTTP yang memanggil API ini
- **`config/services.php`** — konfigurasi URL dan API key (`FORECAST_API_URL`, `FORECAST_API_KEY`)
- **`routes/console.php`** — jadwal prediksi (tiap jam) dan retrain (pukul 02:00)
- **Routes Laravel** — `GET /aqms/dashboard/platform/{uid}/ml-forecast` dan `POST /aqms/dashboard/platform/{uid}/ml-predict`

Service ini **berbagi database** dengan `bc-enviro-web`. Pastikan:

1. `.env` FastAPI (`DB_*`) diisi dengan nilai yang sama dengan `DB_AQMS_*` di Laravel.
2. Nilai `INTERNAL_API_KEY` (FastAPI) **sama persis** dengan `FORECAST_API_KEY` (Laravel).
3. Variabel berikut ditambahkan di `.env` Laravel:

```env
FORECAST_API_URL=http://127.0.0.1:8001
FORECAST_API_KEY=same_as_INTERNAL_API_KEY
```

---

## Production (Systemd)

```ini
[Unit]
Description=Air Quality Forecast API
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/air_quality_forecast
ExecStart=/var/www/air_quality_forecast/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 2
Restart=on-failure
EnvironmentFile=/var/www/air_quality_forecast/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable air-quality-forecast
sudo systemctl start air-quality-forecast
```
