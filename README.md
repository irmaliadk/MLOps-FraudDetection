# MLOps Fraud Detection System

> **Sistem deteksi fraud transaksi crypto real-time yang belajar otomatis dari data baru.**
> Binary classification — setiap transaksi Bitcoin/USD diklasifikasikan sebagai `FRAUD` atau `LEGITIMATE`.

---

## 🏗️ Arsitektur Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MLOPS FRAUD DETECTION PIPELINE                   │
│                                                                         │
│  [Kraken API]                                                           │
│      │  1000 transaksi BTC/USD real-time per batch                      │
│      ▼                                                                  │
│  ┌─────────────────────┐                                                │
│  │  1. DATA INGESTION  │  stream_generator.py                           │
│  │  Ambil + label fraud│  → data/raw/streaming/*.csv                    │
│  └─────────┬───────────┘  Referensi: Kaggle Crypto Scam Dataset         │
│            │  Fraud rate target: 7.25% (berbasis data nyata)            │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  2. PREPROCESSING   │  stream_preprocessor.py                        │
│  │  Scale + engineer   │  → data/processed/streaming/*.csv              │
│  └─────────┬───────────┘  scaler_amount.pkl + scaler_volume.pkl         │
│            │  Timezone: UTC → WIB (Asia/Jakarta)                        │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  3. TRAINING        │  train.py                                      │
│  │  4 model, best F1   │  → MLflow tracking (DagsHub)                   │
│  └─────────┬───────────┘                                                │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  4. QUALITY GATE    │  F1 Score ≥ 0.7?                               │
│  │  Stop jika gagal    │  false → Pipeline STOP                         │
│  └─────────┬───────────┘  true → Lanjut register                        │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  5. MODEL REGISTRY  │  register_model.py                             │
│  │  champion alias     │  → MLflow Registry (DagsHub)                   │
│  └─────────┬───────────┘                                                │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  6. API SERVING     │  FastAPI main.py                               │
│  │  POST /predict      │  → FRAUD / LEGITIMATE + probabilitas           │
│  └─────────┬───────────┘  3 replika Docker (port 8001/8002/8003)        │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  7. MONITORING      │  drift_detector.py + Prometheus + Grafana      │
│  │  Drift + Metrics    │  → reports/drift_result.json                   │
│  └─────────────────────┘                                                │
│            │                                                            │
│            └──────────────────────────────────────────────────────┐     │
│       Auto retrain: tiap push / tiap Minggu / kalau drift ──────► ▲     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🤔 Kenapa MLOps?

| Tanpa MLOps | Dengan MLOps (proyek ini) |
|---|---|
| Training manual tiap ada data baru | Retrain otomatis via 3 trigger GitHub Actions |
| Model tidak terpantau di production | Drift detection KS Test + Prometheus + Grafana |
| Tidak ada versioning model | MLflow Registry dengan alias champion/challenger |
| Tidak bisa rollback model buruk | Challenger = versi lama siap dipakai kembali |
| Deploy manual, error-prone | CI/CD pipeline otomatis end-to-end |
| Data tidak terversioning | DVC + DagsHub remote storage |

---

## 📁 Struktur Direktori

```
MLOps-FraudDetection/
├── .devcontainer/
│   └── devcontainer.json              # GitHub Codespaces config
├── .dvc/
│   └── config                         # DVC remote → DagsHub
├── .github/workflows/
│   ├── mlops-automation.yaml          # CI/CD end-to-end (push/PR/tiap 6 jam)
│   ├── retrain.yml                    # Schedule-based retrain (tiap Minggu)
│   └── drift_trigger.yml              # Drift-based retrain (tiap 12 jam)
├── config/
│   └── model_registry.yaml            # Metadata model aktif
├── data/
│   ├── external/
│   │   └── crypto_scam_transaction_dataset.csv  # Referensi labelling Kaggle
│   ├── raw/streaming/                 # CSV mentah dari Kraken API
│   └── processed/streaming/           # CSV setelah preprocessing
├── models/
│   ├── scalers/
│   │   ├── scaler_amount.pkl          # Scaler untuk fitur amount
│   │   ├── scaler_volume.pkl          # Scaler untuk fitur volume
│   │   └── global_stats.json          # Statistik global akumulatif antar batch
│   ├── trained/
│   │   └── fraud_model.pkl            # Model terbaik hasil training
│   └── registry/                      # Model registry lokal
├── reports/
│   ├── drift_report.html              # Laporan drift (buka di browser)
│   └── drift_result.json              # Hasil KS Test per kolom
├── src/
│   ├── api/
│   │   ├── main.py                    # FastAPI /predict endpoint (production)
│   │   └── serve.py                   # MLflow model serving /invocations
│   ├── dashboard/
│   │   └── app.py                     # Streamlit demo dashboard
│   ├── data/
│   │   ├── reference_labeler.py       # Labelling berbasis pola Kaggle
│   │   ├── simulate_drift.py          # Simulasi data drift
│   │   ├── stream_generator.py        # Fetch data Kraken + label fraud
│   │   └── stream_preprocessor.py     # Clean + scale + feature engineering
│   ├── models/
│   │   ├── evaluate_and_promote.py    # Evaluasi komparatif + promosi model
│   │   ├── register_model.py          # Daftarkan model ke MLflow Registry
│   │   └── train.py                   # Training 4 model + MLflow logging
│   └── monitoring/
│       └── drift_detector.py          # KS Test drift detection
├── tests/
│   └── test_pipeline.py               # Unit tests (pytest)
├── .dockerignore
├── .gitignore
├── Dockerfile                         # Container untuk API service
├── docker-compose.yaml                # Orkestrasi 6 container
├── prometheus.yml                     # Konfigurasi Prometheus scraping
├── prometheus_alerts.yml              # Alert rules Prometheus
└── requirements.txt                   # Python dependencies
```

---

## 🔍 Detail Tiap Komponen

### 1. Data Ingestion — `stream_generator.py`

**Apa yang dilakukan:**
- Hubungi Kraken Public API (tanpa API key)
- Ambil **1000 transaksi** BTC/USD terbaru per batch, **5 batch** per sesi
- Hitung **global stats** (mean & std dari amount/volume) secara akumulatif
- Label fraud menggunakan **referensi pola dari Kaggle Crypto Scam Dataset**
- Simpan ke `data/raw/streaming/XBTUSD_YYYYMMDD_HHMMSS.csv`

**Strategi labelling (bukan self-labeling):**

| Parameter | Nilai | Sumber |
|---|---|---|
| Fraud rate target | 7.25% | Kaggle dataset (20.000 transaksi nyata) |
| Amount fraud ratio | 1.077x | Rata-rata amount fraud vs legitimate |
| Velocity ratio | 1.132x | Rata-rata volume fraud vs legitimate |

**Kenapa global stats penting:**
Threshold fraud tidak dihitung ulang tiap batch. Menggunakan weighted average dari semua batch sebelumnya → label konsisten antar waktu.

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
| Konversi timezone | UTC → WIB (Asia/Jakarta, UTC+7) |
| Ekstrak waktu | `timestamp` → `hour`, `minute` (WIB) |
| Encode side | `s`=sell → `is_sell=1`, `b`=buy → `is_sell=0` |

**Krusial — Training-Serving Consistency:**
Scaler di-fit sekali saat training, lalu **disimpan** ke `models/scalers/scaler_amount.pkl`. Saat inference, scaler yang sama di-load ulang. Kalau berbeda → prediksi salah total (training-serving skew).

**5 fitur final yang masuk ke model:**
`amount_scaled`, `volume_scaled`, `hour`, `minute`, `is_sell`

```bash
python src/data/stream_preprocessor.py
```

---

### 3. Training — `train.py`

**4 model yang dijalankan sekaligus:**

| Model | Parameter | F1 Score |
|---|---|---|
| **RandomForest** ⭐ | n_estimators=100, class_weight=balanced | **0.8660** |
| RandomForest | n_estimators=200, max_depth=10 | 0.8262 |
| DecisionTree | max_depth=10, class_weight=balanced | 0.8401 |
| LogisticRegression | C=0.1, max_iter=1000 | 0.4264 |

**Semua dicatat ke MLflow DagsHub:**
- `mlflow.log_param()` → hyperparameter
- `mlflow.log_metric()` → F1, ROC AUC, Precision, Recall
- `mlflow.sklearn.log_model()` → simpan model artifact

**Kenapa F1, bukan Accuracy?**
Data imbalanced (~7.25% fraud). Model yang selalu prediksi "legitimate" bisa accuracy 92.75% tapi F1=0. F1 mempertimbangkan Precision & Recall sekaligus.

```bash
python src/models/train.py
```

---

### 4. Model Registry — `register_model.py`

**Alur:**
1. Ambil run MLflow dengan F1 tertinggi dari DagsHub
2. Register ke MLflow Model Registry → nama `fraud-detection-best-model`
3. Versi baru dapat alias **`@champion`** (active di production)
4. Versi sebelumnya dapat alias **`@challenger`** (backup, siap rollback)

**Kenapa alias, bukan stage?**
MLflow 2.9+ sudah deprecated stage (Staging/Production). Alias adalah cara yang direkomendasikan.

**Model champion saat ini:**

| Detail | Value |
|---|---|
| Versi | v20 |
| Algoritma | RandomForestClassifier |
| F1 Score | 0.9325 |
| ROC AUC | 0.9495 |
| Precision | 0.9620 |
| Recall | 0.9048 |

```bash
python src/models/register_model.py
```

---

### 5. API Serving — `main.py` & `serve.py`

**Endpoint `main.py` (production, port 8000):**

| Method | URL | Fungsi |
|---|---|---|
| GET | `/` | Info API + model source |
| GET | `/health` | Status API |
| GET | `/metrics` | Prometheus metrics |
| POST | `/predict` | Prediksi fraud |

**Input (nilai RAW, bukan pre-scaled):**
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
  "scaled_values": {"amount_scaled": 1.23, "volume_scaled": 0.45}
}
```

**Endpoint `serve.py` (MLflow serving, port 5001):**
Mensimulasikan `mlflow models serve` dengan endpoint `/invocations`.

```bash
# Production API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# MLflow serving
python src/api/serve.py
```

---

### 6. Drift Detection — `drift_detector.py`

**Cara kerja — Kolmogorov-Smirnov Test:**
- Ambil batch pertama sebagai **reference dataset**
- Ambil batch terbaru sebagai **current dataset**
- Bandingkan distribusi `amount` dan `volume` secara statistik
- Generate `reports/drift_result.json`

**Threshold drift:**

| Parameter | Nilai | Keterangan |
|---|---|---|
| KS p-value | < 0.05 | Distribusi berbeda signifikan |
| Mean shift | > 10% | Rata-rata bergeser lebih dari 10% |
| Drift threshold | ≥ 30% kolom | Minimal 30% kolom drift → retrain |

**Apa itu data drift?**
Pola data di production berubah dari pola saat training. Contoh: market crash Bitcoin → pola transaksi berubah drastis → model lama tidak relevan → perlu retrain otomatis.

```bash
python src/monitoring/drift_detector.py
```

---

### 7. Streamlit Dashboard — `src/dashboard/app.py`

Dashboard demo dengan 4 tab:
- **🎯 Live Prediction** — input transaksi dan prediksi real-time
- **📊 Data Overview** — visualisasi distribusi data streaming
- **🔬 Drift Detection** — status drift dan detail per kolom
- **📈 Model Performance** — perbandingan 4 model eksperimen

```bash
streamlit run src/dashboard/app.py
# Buka: http://localhost:8501
```

---

## 🚀 Cara Menjalankan

### Setup Environment
```bash
# Clone repo
git clone https://github.com/irmaliadk/MLOps-FraudDetection.git
cd MLOps-FraudDetection

# Install dependencies
pip install -r requirements.txt

# Set DagsHub token (untuk MLflow tracking)
export DAGSHUB_TOKEN=your_token_here
```

### Pipeline Manual (urutan wajib)
```bash
python src/data/stream_generator.py      # 1. Fetch 5000 data dari Kraken
python src/data/stream_preprocessor.py  # 2. Preprocessing + simpan scaler
python src/models/train.py               # 3. Train 4 model + log ke DagsHub
python src/models/register_model.py     # 4. Register model terbaik
python src/monitoring/drift_detector.py # 5. Cek data drift
```

### Jalankan API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Test: http://localhost:8000/docs
```

### Jalankan Streamlit Dashboard
```bash
streamlit run src/dashboard/app.py
# Buka: http://localhost:8501
```

### Jalankan Docker Compose (6 container)
```bash
docker compose up -d
docker compose ps
```

**Port yang tersedia:**

| Service | Port | URL |
|---|---|---|
| API Replica 1 | 8001 | http://localhost:8001 |
| API Replica 2 | 8002 | http://localhost:8002 |
| API Replica 3 | 8003 | http://localhost:8003 |
| MLflow Server | 5000 | http://localhost:5000 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 (admin/admin123) |

### Simulasi Data Drift
```bash
python src/data/simulate_drift.py        # Generate shifted data
python src/data/stream_preprocessor.py  # Preprocess ulang
python src/monitoring/drift_detector.py # Deteksi drift
```

---

## ⚙️ CI/CD — GitHub Actions

### 3 Workflow Independen

**1. `mlops-automation.yaml` — Full Pipeline**
Trigger: Push ke `main`, Pull Request, atau tiap 6 jam

```
Push/PR/Cron → Fetch data → Preprocess → Test → Train
             → Eval (F1 ≥ 0.7?) → Register → Commit
```

**2. `retrain.yml` — Weekly Retrain**
Trigger: Setiap Minggu jam 00:00 UTC

```
Cron Minggu → Fetch data baru → Preprocess → Train → Evaluate & Promote
```

**3. `drift_trigger.yml` — Drift-based Retrain**
Trigger: Setiap 12 jam (hanya retrain kalau drift terdeteksi!)

```
Cron 12 jam → Fetch data → Preprocess → Cek drift
            → Drift? YES → Train → Evaluate & Promote
            → Drift? NO  → Skip (model masih relevan)
```

---

## 📊 Observability

### Prometheus + Grafana
```bash
docker compose up -d
# Grafana: http://localhost:3000 (admin/admin123)
# Prometheus: http://localhost:9090
```

**Metrics yang dipantau:**
- Request Rate ke `/predict`
- Average Latency endpoint `/predict`
- Total Requests per endpoint

### Alert Rules (prometheus_alerts.yml)
- Latency > 500ms selama 2 menit → warning
- Error rate > 10 requests/menit → critical
- Tidak ada request 5 menit → warning

---

## 🗃️ Data Versioning — DVC + DagsHub

```bash
dvc status          # Cek status data
dvc diff            # Lihat perubahan versi
dvc push            # Upload ke DagsHub
dvc pull            # Download dari DagsHub
```

---

## 🔗 Links Penting

| Resource | URL |
|---|---|
| **MLflow DagsHub** | https://dagshub.com/irmaliadk/MLOps-FraudDetection.mlflow |
| **DVC Remote** | https://dagshub.com/irmaliadk/MLOps-FraudDetection.dvc |
| **API Docs** | http://localhost:8000/docs |
| **Streamlit Dashboard** | http://localhost:8501 |
| **Grafana** | http://localhost:3000 |

---

## 🛠️ Tech Stack

| Komponen | Tools |
|---|---|
| **Sumber Data** | Kraken Public API (XBTUSD) |
| **Referensi Label** | Kaggle Crypto Scam Transaction Dataset |
| **ML Framework** | scikit-learn |
| **Experiment Tracking** | MLflow + DagsHub |
| **Model Registry** | MLflow Registry (@champion/@challenger) |
| **Data Versioning** | DVC + DagsHub |
| **API Serving** | FastAPI + Uvicorn |
| **Model Serving** | MLflow pyfunc |
| **Drift Detection** | KS Test (scipy) |
| **Monitoring** | Prometheus + Grafana |
| **CI/CD** | GitHub Actions (3 workflows) |
| **Containerisasi** | Docker + Docker Compose (6 container) |
| **Demo Dashboard** | Streamlit + Plotly |
| **Dev Environment** | GitHub Codespaces |

---

## 📋 Quick Commands Cheatsheet

```bash
# ===== PIPELINE MANUAL =====
python src/data/stream_generator.py          # Fetch 5000 data Kraken
python src/data/stream_preprocessor.py       # Preprocess + scaler
python src/models/train.py                   # Train 4 model
python src/models/register_model.py          # Register champion
python src/monitoring/drift_detector.py      # Cek drift

# ===== API =====
uvicorn src.api.main:app --port 8000         # Production API
python src/api/serve.py                      # MLflow serving

# ===== DASHBOARD =====
streamlit run src/dashboard/app.py           # Demo dashboard

# ===== MLFLOW UI =====
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ===== DOCKER =====
docker compose up -d                         # Jalankan semua
docker compose ps                            # Cek status
docker compose down                          # Stop semua

# ===== DVC =====
dvc status && dvc diff                       # Cek versioning data

# ===== TESTS =====
pytest tests/ -v                             # Unit tests

# ===== DRIFT SIMULATION =====
python src/data/simulate_drift.py && \
python src/data/stream_preprocessor.py && \
python src/monitoring/drift_detector.py
```

---

## 🌿 Branching Strategy

Proyek menggunakan **GitHub Flow**:
- `main` — branch production, hanya menerima merge dari Pull Request
- `feat/*` — branch untuk fitur baru
- `fix/*` — branch untuk perbaikan bug