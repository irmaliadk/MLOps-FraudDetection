# MLOps Fraud Detection System

> **Sistem deteksi fraud transaksi crypto real-time yang belajar otomatis dari data baru.**
> Binary classification — setiap transaksi Bitcoin/USD diklasifikasikan sebagai `FRAUD` atau `LEGITIMATE`.

---

## Arsitektur Pipeline (Big Picture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MLOPS FRAUD DETECTION PIPELINE                   │
│                                                                         │
│  [Kraken API]                                                           │
│      │  100 transaksi BTC/USD real-time                                 │
│      ▼                                                                  │
│  ┌─────────────────────┐                                                │
│  │  1. DATA INGESTION  │  stream_generator.py                           │
│  │  Ambil + label fraud│  → data/raw/streaming/*.csv                    │
│  └─────────┬───────────┘                                                │
│            │  amount/volume > mean + 1 std → Class=1 (FRAUD)            │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  2. PREPROCESSING   │  stream_preprocessor.py                        │
│  │  Scale + engineer   │  → data/processed/streaming/*.csv              │
│  └─────────┬───────────┘  scaler_amount.pkl + scaler_volume.pkl         │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  3. TRAINING        │  train.py                                      │
│  │  4 model, best F1   │  → MLflow tracking (mlflow.db)                 │
│  └─────────┬───────────┘                                                │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  4. QUALITY GATE    │  F1 Score > 0.7?                               │
│  │  Stop jika gagal    │  false → Pipeline berhenti                     │
│  └─────────┬───────────┘  true → Lanjut register                        │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  5. MODEL REGISTRY  │  register_model.py                             │
│  │  champion alias     │  → MLflow Model Registry                       │
│  └─────────┬───────────┘                                                │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  6. API SERVING     │  FastAPI main.py                               │
│  │  POST /predict      │  → FRAUD / LEGITIMATE + probabilitas           │
│  └─────────┬───────────┘  3 replika Docker                              │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  7. MONITORING      │  drift_detector.py                             │
│  │  Data drift check   │  → reports/drift_report.html                   │
│  └─────────────────────┘                                                │
│            │                                                            │
│            └──────────────────────────────────────────────────────┐     │
│                          Auto retrain (GitHub Actions)            │     │
│                          tiap push / tiap minggu ──────────────►  ▲     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kenapa MLOps?

| Tanpa MLOps | Dengan MLOps (proyek ini) |
|---|---|
| Training manual tiap ada data baru | Retrain otomatis tiap minggu via GitHub Actions |
| Model tidak terpantau di production | Drift detection dengan Evidently AI |
| Tidak ada versioning model | MLflow Registry dengan alias champion/challenger |
| Tidak bisa rollback model buruk | Challenger = versi lama siap dipakai kembali |
| Deploy manual, error-prone | CI/CD pipeline otomatis end-to-end |

---

## Struktur Direktori

```
MLOps-FraudDetection/
├── .devcontainer/
│   └── devcontainer.json          # GitHub Codespaces config
├── .dvc/
│   └── config                     # DVC remote → DagsHub
├── .github/workflows/
│   ├── mlops-automation.yaml      # CI/CD end-to-end (push/tiap 6 jam)
│   └── retrain.yml                # Weekly retrain otomatis
├── config/
│   └── model_registry.yaml        # Metadata model aktif
├── data/
│   ├── raw/streaming/             # CSV mentah dari Kraken API
│   └── processed/streaming/       # CSV setelah preprocessing
├── models/
│   ├── trained/fraud_model.pkl    # Model terbaik (best F1)
│   └── scalers/                   # scaler_amount.pkl + scaler_volume.pkl
├── reports/
│   └── drift_report.html          # Laporan drift (buka di browser)
├── src/
│   ├── api/
│   │   ├── main.py                # FastAPI /predict endpoint
│   │   └── serve.py               # MLflow serve /invocations
│   ├── data/
│   │   ├── stream_generator.py    # Ambil data Kraken + label fraud
│   │   └── stream_preprocessor.py # Clean + scale + feature engineering
│   ├── models/
│   │   ├── train.py               # Training 4 model + MLflow logging
│   │   └── register_model.py      # Daftarkan model ke registry
│   └── monitoring/
│       └── drift_detector.py      # Deteksi data drift
├── tests/
│   └── test_pipeline.py           # Unit tests (pytest)
├── docker-compose.yaml            # 3 replika api-service + mlflow-server
├── Dockerfile
└── requirements.txt
```

---

## Detail Tiap Komponen

### 1. Data Ingestion — `stream_generator.py`

**Apa yang dilakukan:**
- Hubungi Kraken Public API (tanpa API key)
- Ambil 100 transaksi BTC/USD terbaru
- Hitung **global stats** (mean & std dari amount/volume) — diakumulasi, bukan di-reset
- Label fraud: `amount > mean + 1*std` ATAU `volume > mean + 1*std` → `Class=1`
- Simpan ke `data/raw/streaming/XBTUSD_YYYYMMDD_HHMMSS.csv`

**Kenapa global stats penting:**
Threshold fraud tidak dihitung ulang tiap batch. Digunakan weighted average dari semua batch sebelumnya → label konsisten antar waktu.

```bash
python src/data/stream_generator.py
```

---

### 2. Preprocessing — `stream_preprocessor.py`

**Transformasi yang dilakukan:**

| Langkah | Detail |
|---|---|
| Hapus missing values | `df.dropna()` |
| Hapus duplikat | `df.drop_duplicates()` |
| Scale amount | `StandardScaler` → `amount_scaled` |
| Scale volume | `StandardScaler` → `volume_scaled` |
| Ekstrak waktu | `timestamp` → `hour`, `minute` |
| Encode side | `s` = sell → `is_sell=1`, `b` = buy → `is_sell=0` |

**Krusial:** Scaler di-fit sekali saat training, lalu **disimpan** ke `models/scalers/scaler_amount.pkl`. Saat inference, scaler yang sama di-load ulang. Kalau berbeda → prediksi salah total.

**5 fitur final yang masuk ke model:**
`amount_scaled`, `volume_scaled`, `hour`, `minute`, `is_sell`

```bash
python src/data/stream_preprocessor.py
```

---

### 3. Training — `train.py`

**4 model yang dijalankan sekaligus:**

| Model | Parameter |
|---|---|
| RandomForest | n_estimators=100 |
| RandomForest | n_estimators=200, max_depth=10 |
| DecisionTree | max_depth=10 |
| LogisticRegression | C=0.1 |

**Semua dicatat ke MLflow:**
- `mlflow.log_param()` → hyperparameter
- `mlflow.log_metric()` → F1, AUC, precision, recall
- `mlflow.sklearn.log_model()` → simpan model artifact

**Split data:** 80% train / 20% test, `stratify=y` (proporsi fraud seimbang)

**Kenapa F1, bukan Accuracy?**
Data sangat imbalanced. Model yang selalu prediksi "legitimate" pun bisa accuracy 95% tapi F1=0. F1 mempertimbangkan precision & recall sekaligus.

```bash
python src/models/train.py
```

---

### 4. Model Registry — `register_model.py`

**Alur:**
1. Ambil run MLflow dengan F1 tertinggi
2. Register ke MLflow Model Registry → nama `fraud-detection-best-model`
3. Versi baru dapat alias **`champion`** (active di production)
4. Versi sebelumnya dapat alias **`challenger`** (backup, siap rollback)

**Kenapa alias, bukan stage?**
MLflow 2.9+ sudah deprecated stage (Staging/Production). Alias adalah cara yang direkomendasikan sekarang.

```bash
python src/models/register_model.py
```

---

### 5. API Serving — `main.py`

**Endpoint:**

| Method | URL | Fungsi |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Status API + info model |
| POST | `/predict` | Prediksi fraud |

**Input (nilai RAW, bukan scaled):**
```json
{
  "is_sell": 1,
  "amount": 103500.25,
  "volume": 0.00234,
  "hour": 14,
  "minute": 30
}
```

**Output:**
```json
{
  "prediction": 1,
  "label": "FRAUD",
  "fraud_probability": 0.87,
  "model_version": "champion"
}
```

**Load model:** Prioritas dari MLflow Registry `@champion`, fallback ke `.pkl` lokal jika registry tidak tersedia.

```bash
# Jalankan API lokal
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test prediksi
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"is_sell": 1, "amount": 103500.25, "volume": 0.00234, "hour": 14, "minute": 30}'
```

---

### 6. Monitoring Drift — `drift_detector.py`

**Cara kerja:**
- Ambil batch pertama sebagai **reference dataset**
- Ambil batch terbaru sebagai **current dataset**
- Bandingkan distribusi fitur menggunakan **Evidently AI**
- Generate laporan HTML ke `reports/drift_report.html`

**Apa itu data drift?**
Pola data di production berubah dari pola saat training. Contoh: market crash → pola transaksi BTC berubah drastis → model lama tidak relevan → perlu retrain.

```bash
python src/monitoring/drift_detector.py
# Buka laporan:
open reports/drift_report.html
```

---

### 7. Data Versioning — DVC

DVC memisahkan versioning **kode** (Git) dari versioning **data** (DVC remote → DagsHub).

```bash
# Setelah fetch data baru:
dvc add data/raw/streaming/
git add .
git commit -m "data: add new batch YYYYMMDD"
git push

# Cek perubahan versi data:
dvc status
dvc diff
```

---

## Docker Compose

**Arsitektur container:**

```
┌─────────────────────────────────────┐
│          mlops-network              │
│                                     │
│  ┌──────────────────┐               │
│  │  mlflow-server   │ :5000         │
│  │  SQLite backend  │               │
│  └────────┬─────────┘               │
│           │ depends_on              │
│  ┌────────┴────────────────────┐    │
│  │  api-service (3 replicas)   │    │
│  │  :8000 / :8001 / :8002      │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

```bash
# Jalankan semua service
docker compose up -d

# Cek status replika (harus ada 3 api-service)
docker compose ps

# Scale manual
docker compose up -d --scale api-service=5

# Stop semua
docker compose down
```

**Kenapa 3 replika?** Horizontal scaling — beban request dibagi ke 3 container. Kalau satu mati, dua yang lain masih jalan (high availability).

---

## ⚙️ CI/CD — GitHub Actions

### `mlops-automation.yaml` — Full Pipeline
**Trigger:** Push ke `main`, Pull Request, atau tiap 6 jam (cron)

```
Push/PR/Cron
     │
     ▼
1. Fetch data (stream_generator.py)
     │
     ▼
2. Preprocessing (stream_preprocessor.py)
     │
     ▼
3. Unit tests (pytest)
     │
     ▼
4. Training (train.py)
     │
     ▼
5. Cek F1 > 0.7 ──── Gagal → pipeline STOP (model tidak di-deploy)
     │ Lolos
     ▼
6. Register model (register_model.py) → champion alias
```

### `retrain.yml` — Weekly Retrain
**Trigger:** Setiap Minggu jam 00:00 atau manual dispatch

```
Setiap Minggu
     │
     ▼
1. Fetch data baru
2. Preprocessing
3. Training ulang
4. Cek data drift
```

---

## Demo Flow

### Step 1 — Tampilkan MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Buka: http://localhost:5000
```
Yang ditunjukkan: list run, grafik F1 per model, parameter tiap eksperimen.

### Step 2 — Jalankan API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3 — Demo Prediksi (transaksi normal)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"is_sell": 0, "amount": 95000.0, "volume": 0.001, "hour": 10, "minute": 15}'
```
Expected: `LEGITIMATE`

### Step 4 — Demo Prediksi (transaksi mencurigakan)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"is_sell": 1, "amount": 500000.0, "volume": 99.9, "hour": 3, "minute": 47}'
```
Expected: `FRAUD`

### Step 5 — Tampilkan Docker Compose
```bash
docker compose up -d
docker compose ps
# Tunjukkan 3 replika api-service berjalan
```

### Step 6 — Tampilkan Drift Report
```bash
open reports/drift_report.html
```

### Step 7 — Tampilkan GitHub Actions
Buka tab **Actions** di GitHub repo → tunjukkan riwayat pipeline yang pernah jalan.

---

## 🛠️ Tech Stack

| Komponen | Tools | Fungsi |
|---|---|---|
| Sumber data | Kraken Public API | 100 transaksi BTC/USD real-time |
| ML framework | scikit-learn | Training model |
| Experiment tracking | MLflow | Log params, metrics, model artifact |
| Model registry | MLflow Registry | Versioning + alias champion/challenger |
| API serving | FastAPI | Endpoint `/predict` |
| Drift detection | Evidently AI | Laporan HTML perubahan distribusi data |
| Data versioning | DVC + DagsHub | Versioning data terpisah dari Git |
| Containerisasi | Docker + Docker Compose | 3 replika API + MLflow server |
| CI/CD | GitHub Actions | Pipeline otomatis end-to-end |
| Dev environment | GitHub Codespaces | Reproducible dev environment |

---

## 📋 Quick Commands Cheatsheet

```bash
# ===== PIPELINE MANUAL =====
python src/data/stream_generator.py          # 1. Fetch data
python src/data/stream_preprocessor.py       # 2. Preprocess
python src/models/train.py                   # 3. Train
python src/models/register_model.py          # 4. Register model
python src/monitoring/drift_detector.py      # 5. Cek drift

# ===== API =====
uvicorn src.api.main:app --port 8000         # Jalankan API
curl http://localhost:8000/health            # Health check

# ===== MLFLOW UI =====
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ===== DOCKER =====
docker compose up -d                         # Jalankan semua
docker compose ps                            # Cek status replika
docker compose down                          # Stop semua

# ===== DVC =====
dvc status                                   # Cek status data
dvc diff                                     # Lihat perubahan versi

# ===== TESTS =====
pytest tests/ -v                             # Jalankan unit tests
```

---

## Links

- **MLflow UI:** http://localhost:5000
- **API Docs (Swagger):** http://localhost:8000/docs
- **Drift Report:** `reports/drift_report.html`
- **DVC Remote:** DagsHub
- **CI/CD:** GitHub Actions tab di repositori

---
