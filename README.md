# Air Quality Forecast API

FastAPI service yang menyediakan prakiraan kualitas udara 6 jam ke depan per sensor, lengkap dengan pita ketidakpastian P10–P90 yang cakupannya teruji. Modelnya **LightGBM langsung multi-horizon** pada fitur lag dan siklus harian. Dirancang untuk diintegrasikan dengan sistem monitoring AQMS berbasis Laravel.

---

## Arsitektur Model

```
t_loggers (per menit)
      │  rata-rata per jam di MySQL; jam < 10 bacaan dan nilai di luar batas
      │  fisik dikosongkan — TIDAK ada interpolasi
      ▼
Deret per jam (NaN = tidak terukur)
      │  log1p untuk konsentrasi & indeks
      ▼
Fitur per (jam asal t, horizon h), semuanya relatif terhadap nilai terakhir:
      │   • tren & sebaran 1 jam – 7 hari terakhir
      │   • nilai pada JAM YANG SAMA kemarin, 2 hari lalu, sepekan lalu
      │   • parameter lain sebagai konteks (PM, kelembapan, suhu, kebisingan)
      │   • jam lokal (WITA) dari jam target, hari dalam pekan, horizon
      ▼
LightGBM × 3 per parameter      titik (median, L1) · P10 · P90
      │  target: SELISIH terhadap nilai terakhir, di ruang log
      ▼
Kalibrasi dari prakiraan luar-sampel
      │   • penyusutan α menuju persistence (per horizon)
      │   • pelebaran interval konformal (per horizon)
      ▼
Prakiraan 6 langkah (1 jam/langkah) + batas bawah/atas
```

Setiap sensor (`uid`) punya satu berkas model `saved_models/{uid}/forecaster.pkl` dan satu laporan evaluasi `saved_models/{uid}/backtest.json`.

### Dasar pemilihan metode

Metode ini menggantikan SSA-BiLSTM-LightGBM. Keduanya diuji pada data yang sama: ekspor AQMS 7 Okt 2025 – 3 Nov 2026, 9 sensor dengan riwayat setahun penuh, prakiraan dinilai dalam µg/m³ terhadap rata-rata jam teramati.

**Uji langsung, baris yang sama.** Lipatan uji 6 Okt – 3 Nov 2026, 9 sensor, 35.959 pasangan (jam asal, horizon) per parameter. Kedua metode dilatih hanya pada data sebelum 6 Okt. MAE dalam µg/m³; dalam kurung, skill terhadap persistence (1 − MAE/MAE persistence).

| Parameter | Persistence | Jam yang sama kemarin | SSA-BiLSTM-LightGBM | LightGBM langsung |
|---|---|---|---|---|
| TSP | 36,94 | 41,89 | 31,70 (14,2 %) | **27,65 (25,1 %)** |
| PM10 | 10,17 | 15,61 | 10,10 (0,8 %) | **9,01 (11,4 %)** |
| PM2.5 | 6,96 | 10,75 | 6,91 (0,6 %) | **6,23 (10,5 %)** |

- Metode lama **kalah dari persistence di 4 dari 9 sensor** pada TSP (terburuk −100 %), walau sudah dilindungi penyusutan α: bobot yang dicari pada satu periode kalibrasi tidak menjamin apa pun pada periode berikutnya, dan keempat sensor itu kadarnya jauh lebih rendah pada bulan uji. Metode baru lebih baik daripada metode lama di kesembilan sensor (4–48 %) dan di keenam horizon (4–17 %); ia sendiri sedikit di bawah persistence di 2 sensor yang bulan itu kadarnya sangat rendah (−3 % dan −4 %).
- Diulang pada musim lain (16 Jun – 14 Jul 2026, 3 sensor, 12.090 pasangan) hasilnya searah: MAE TSP 35,58 → 30,05 (skill 16,3 % → 29,3 %), PM10 12,13 → 10,58, PM2.5 5,00 → 4,59. Di sini metode lama kalah telak dari persistence di satu sensor (−105 %); di satu sensor lain keduanya praktis seri (metode baru 1,8 % lebih buruk pada TSP). Dari 12 pasangan sensor–lipatan yang diuji, metode baru unggul pada TSP di 11.
- Cakupan pita P10–P90 untuk TSP (nominal 80 %): metode lama 77 %, turun ke 70 % di H+6, 64–87 % antarsensor. Metode baru 84 %, 82–88 % antarhorizon, 80–90 % antarsensor.
- Waktu latih per sensor di mesin uji (Apple M4): metode lama 9–22 menit (tiga pelatihan berjalan bersamaan); metode baru ±2 menit dengan 4 thread, sudah termasuk backtest enam lipatan.

Hal lain yang diperbaiki sekaligus:

| Sebelumnya | Sekarang |
|---|---|
| Celah data (bahkan berhari-hari) diisi garis lurus lalu dilatihkan | Jam tak terukur dibiarkan kosong; tidak menjadi target, jangkar, maupun skor |
| LightGBM dilatih pada prediksi BiLSTM in-sample (stacking bocor) | Satu tahap; tidak ada stacking |
| Skor di ruang log-MinMax, rata-rata 14 variabel, satu potongan uji ±2 minggu | MAE/RMSE/bias per parameter per horizon dalam satuan fisik, rolling-origin beberapa lipatan 4-mingguan |
| Hanya dibandingkan dengan persistence | Juga persistence musiman (jam yang sama kemarin) + uji Diebold–Mariano |
| P10–P90 tanpa uji cakupan | Interval dikalibrasi konformal; cakupan dilaporkan |
| Sensor mati tetap "diramal" dari data lama | Bacaan terakhir lebih tua dari 3 jam → tidak ada prakiraan untuk parameter itu |
| Sensor tanpa modul cuaca mendapat prakiraan suhu/tekanan/kelembapan karangan (28 °C / 760 / 82 %) | Parameter tanpa data bernilai `null` |
| TensorFlow, ±800 MB artefak | Tanpa TensorFlow; ±15 MB per sensor |

Yang sengaja **tidak** dipakai: prakiraan cuaca NWP (Open-Meteo) untuk jam target. Diuji pada data yang sama, tambahan angin/hujan/tinggi lapisan batas hanya menurunkan MAE TSP 0,8 % — itu pun batas atas, karena data historisnya berkualitas analisis, bukan prakiraan sungguhan. Tidak sepadan dengan ketergantungan pada layanan luar di setiap prediksi. Data cuaca stasiun tambang, bila kelak tersedia, layak diuji ulang dengan `scripts/backtest.py`.

### Keterbatasan yang perlu diketahui

- Prakiraan titik adalah **median**, bukan rata-rata. Untuk konsentrasi yang condong-kanan, median berada di bawah rata-rata, sehingga bias rata-ratanya negatif (lihat `bias` di laporan). Untuk peringatan dini gunakan batas atas (P90), bukan titiknya.
- Lonjakan mendadak tetap sulit diramal dari riwayat sensor saja; lihat blok `exceedance` (POD/FAR/CSI) di laporan untuk kemampuan deteksi sebenarnya.
- Ambang pada blok `exceedance` adalah baku mutu **24 jam** PP 22/2021 yang diterapkan pada nilai per jam — indikator peringatan, bukan penilaian kepatuhan.
- Indeks (`aqi_index*`) diramal sebagai deret tersendiri, belum diturunkan dari konsentrasi prakiraan.

---

## Struktur Proyek

```
air_quality_forecast/
├── app/
│   ├── auth.py              # API key dependency (require_api_key)
│   ├── config.py            # ENV vars, konstanta model, path
│   ├── db.py                # MySQL: simpan/baca prediksi & metadata
│   ├── main.py              # FastAPI app entry point + penjadwal per jam
│   ├── models/
│   │   ├── preprocessor.py  # t_loggers → deret per jam tersaring (tanpa interpolasi)
│   │   ├── features.py      # Fitur (jam asal, horizon) dan transformasi log
│   │   ├── forecaster.py    # LightGBM titik/P10/P90, simpan & muat model
│   │   ├── calibration.py   # Penyusutan α dan interval konformal
│   │   ├── backtest.py      # Rolling-origin, metrik, uji Diebold–Mariano
│   │   ├── pipeline.py      # Alur latih (fit_sensor) dan prediksi
│   │   └── tuning.py        # Optuna untuk hiperparameter LightGBM
│   ├── routers/             # prediction, training, accuracy, dispersion
│   └── services/            # predict, retrain, accuracy, dispersion
├── scripts/
│   ├── init_db.py           # Buat tabel predictions & model_metadata
│   └── backtest.py          # Evaluasi semua sensor → JSON + CSV untuk naskah
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

### Penjadwal prediksi

Layanan memprediksi semua sensor sendiri **tiap jam pada menit tetap** (bawaan `:55`, saat ember jam berjalan sudah hampir lengkap), lalu memperbarui nilai aktual dan cek drift. Atur lewat `.env`:

| Variable | Bawaan | Keterangan |
|---|---|---|
| `PREDICT_SCHEDULE_MINUTE` | `55` | Menit pada setiap jam (0–59) |
| `INTERNAL_SCHEDULER` | `1` | `0` mematikan penjadwal internal, mis. bila prediksi dipicu dari luar (`POST /predict/all`) |

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
| POST | `/tune/{uid}` | API Key | Setel hiperparameter (Optuna) lalu retrain (async, 202) |
| GET | `/training/progress/{uid}` | API Key | Fase dan persentase pelatihan yang sedang berjalan |
| GET | `/training/report/{uid}` | API Key | Laporan backtest pelatihan terakhir |

**Autentikasi:** sertakan header `X-API-Key: <INTERNAL_API_KEY>` untuk endpoint yang membutuhkan auth. Jika `INTERNAL_API_KEY` kosong di `.env`, autentikasi dinonaktifkan.

---

## Format Response Prediksi

```json
{
  "uid": "AQI01B",
  "predictions": [
    {
      "step": 1,
      "target_time": "2026-11-16T03:00:00",
      "pm_25": 12.34,
      "pm_10": 25.67,
      "tsp": 30.12,
      "noise": 51.8,
      "temp": null,
      "humidity": null,
      "mmhg": null,
      "aqi_index": 45.0,
      "lower_bounds": { "tsp": 18.4, "pm_10": 15.1, "temp": null },
      "upper_bounds": { "tsp": 61.7, "pm_10": 44.9, "temp": null }
    }
    // ... step 2–6
  ]
}
```

- `target_time` adalah awal jam (UTC). Langkah 1 adalah jam yang sedang berjalan — jam pertama yang belum teramati penuh.
- `lower_bounds`/`upper_bounds` adalah P10 dan P90 terkalibrasi (cakupan nominal 80 %).
- Nilai `null` berarti parameter itu tidak dimodelkan untuk sensor ini (mis. suhu pada sensor tanpa modul cuaca) atau bacaan terakhirnya lebih tua dari 3 jam.

---

## Alur Kerja

### Pertama Kali (Training)

Model harus dilatih dulu; `/predict` pada sensor yang belum dilatih menjawab `Model not trained yet — run /retrain first`.

```bash
curl -X POST http://localhost:8001/retrain/AQI01B -H "X-API-Key: your_api_key"
curl http://localhost:8001/training/progress/AQI01B -H "X-API-Key: your_api_key"
```

Butuh minimal 20 hari data per jam. Satu sensor dengan data setahun selesai dalam hitungan menit.

### Retrain Terjadwal

Retrain dijadwalkan dari sisi Laravel (via `routes/console.php`) setiap hari pukul 02:00. Endpoint retrain langsung merespons `202 Accepted` karena prosesnya berjalan di background. Retrain juga terpicu otomatis bila galat TSP 24 jam terakhir melebihi 1,5× galat historisnya.

### Prediksi On-Demand

```bash
# Prediksi satu sensor
curl -X POST http://localhost:8001/predict/AQI01B \
     -H "X-API-Key: your_api_key"

# Ambil hasil terakhir (tanpa auth)
curl http://localhost:8001/predictions/AQI01B
```

### Laporan Evaluasi

Setiap retrain menjalankan backtest rolling-origin dan menyimpan hasilnya. Lipatan pertama hanya untuk kalibrasi; skor dihitung dari lipatan berikutnya, masing-masing dengan model yang hanya dilatih pada data sebelum lipatan itu dan kalibrasi dari lipatan-lipatan sebelumnya.

```bash
curl http://localhost:8001/training/report/AQI01B -H "X-API-Key: your_api_key"
```

Per parameter dan per horizon, dalam satuan fisik: `mae`, `rmse`, `bias`, `mae_persistence`, `mae_seasonal_naive`, `skill_vs_persistence`, `skill_vs_seasonal_naive`, `coverage` (cakupan P10–P90), `interval_width_median`, `dm_pvalue_vs_persistence`, serta blok `exceedance` (POD/FAR/CSI) untuk TSP, PM10, dan PM2.5.

Untuk tabel naskah — semua sensor sekaligus, tanpa menyentuh model produksi:

```bash
python scripts/backtest.py                        # dari basis data
python scripts/backtest.py --csv-dir /path/ekspor # dari CSV per menit
# → logs/backtest/ringkasan.csv dan logs/backtest/<uid>.json
```

> Kolom `model_metadata.mae_score` (lencana "% acc" di dasbor) adalah MAE tak bersatuan yang dinormalkan terhadap rentang data, dipertahankan agar tampilan tidak berubah. Angka itu bukan untuk dilaporkan; pakai laporan di atas.

### Migrasi dari versi SSA-BiLSTM-LightGBM

1. `pip install -r requirements.txt` — TensorFlow tidak lagi dibutuhkan (`pip uninstall tensorflow` bila ingin menghemat ruang).
2. Restart service, lalu `POST /retrain/all`. Sampai sebuah sensor selesai dilatih ulang, `/predict` untuk sensor itu menjawab "belum dilatih".
3. Hapus artefak lama: `saved_models/*/bilstm.keras`, `lgbm*.pkl`, `scaler.pkl`, `shrinkage.json`. `best_params.json` lama diabaikan otomatis; jalankan `/tune/{uid}` bila ingin menyetel ulang.
4. Baris `predictions` sebelum migrasi berasal dari model lama; kecualikan dari analisis akurasi model baru.

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
