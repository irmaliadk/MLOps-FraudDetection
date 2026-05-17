# MLOps Fraud Detection System

Sistem deteksi fraud pada transaksi keuangan yang dibangun dengan pendekatan MLOps,
mencakup continual learning, data versioning, dan automated retraining pipeline.

## Tujuan Proyek

Membangun sistem ML production-ready yang mampu:
- Mendeteksi transaksi fraud secara real-time
- Beradaptasi terhadap perubahan pola fraud (data drift)
- Melakukan retraining model secara otomatis dan terjadwal

## ML Task

Binary Classification — setiap transaksi diklasifikasikan sebagai:
- `0` : Transaksi legitimate
- `1` : Transaksi fraud

## Sumber Data

Menggunakan **Kraken Public API** — data transaksi crypto XBTUSD
yang diambil secara real-time setiap 30 detik tanpa memerlukan API key.
Label fraud ditentukan secara otomatis berdasarkan rules statistik:
transaksi dengan amount atau volume di atas 2 standar deviasi dari
rata-rata diklasifikasikan sebagai fraud.

## Struktur Direktori

```
MLOps-FraudDetection/
├── .devcontainer/
│   └── devcontainer.json          # Konfigurasi GitHub Codespaces
├── .dvc/
│   ├── config                     # Konfigurasi DVC remote (DagsHub)
│   └── .gitignore
├── .github/
│   └── workflows/
│       ├── mlops-automation.yaml  # End-to-end CI/CD pipeline
│       └── retrain.yml            # Weekly retrain otomatis
├── config/
│   └── model_registry.yaml        # Metadata model aktif
├── data/
│   ├── raw/
│   │   └── streaming/             # Data mentah real-time dari Kraken API
│   ├── processed/
│   │   └── streaming/             # Data setelah preprocessing
│   └── external/                  # Data referensi eksternal
├── models/
│   ├── trained/
│   │   └── fraud_model.pkl        # Model terbaik hasil training
│   └── registry/                  # Model registry lokal
├── mlruns/                        # Artifact MLflow experiment tracking
├── notebooks/                     # Jupyter notebooks eksplorasi
├── reports/
│   └── drift_report.html          # Laporan drift detection Evidently
├── src/
│   ├── api/
│   │   ├── main.py                # FastAPI inference endpoint
│   │   └── serve.py               # MLflow model serving script
│   ├── data/
│   │   ├── ingest.py              # Script ingestion data Kraken
│   │   ├── stream_generator.py    # Generator streaming data real-time
│   │   └── stream_preprocessor.py # Preprocessing data streaming
│   ├── features/
│   │   └── build_features.py      # Feature engineering
│   ├── models/
│   │   ├── train.py               # Script training & MLflow logging
│   │   └── register_model.py      # Script registrasi model ke MLflow Registry
│   ├── monitoring/
│   │   └── drift_detector.py      # Deteksi data drift dengan Evidently
│   ├── ingest_data.py             # Entry point ingestion data
│   └── preprocess.py              # Entry point preprocessing
├── tests/
│   └── test_pipeline.py           # Unit tests dengan pytest
├── .dvcignore
├── .gitignore
├── Dockerfile                     # Container untuk API service
├── docker-compose.yaml            # Orkestrasi multi-container
├── mlflow.db                      # Database MLflow lokal
├── requirements.txt               # Python dependencies
└── README.md
```

## Cara Menjalankan dengan GitHub Codespaces

1. Buka repositori ini di GitHub
2. Klik tombol hijau **"Code"**
3. Pilih tab **"Codespaces"**
4. Klik **"Create codespace on main"**
5. Tunggu environment selesai dibangun
6. Selesai — environment siap digunakan

## Cara Menjalankan Data Ingestion & Preprocessing

### 1. Fetch data streaming dari Kraken
```bash
python src/data/stream_generator.py
```
Output: file baru di `data/raw/streaming/XBTUSD_YYYYMMDD_HHMMSS.csv`

### 2. Preprocessing data streaming
```bash
python src/data/stream_preprocessor.py
```
Output: file baru di `data/processed/streaming/processed_YYYYMMDD_HHMMSS.csv`

### 3. Training model
```bash
python src/models/train.py
```

### 4. Register model terbaik ke MLflow Registry
```bash
python src/models/register_model.py
```

### 5. Cek drift
```bash
python src/monitoring/drift_detector.py
```

### 6. Jalankan API inference
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Model Serving & Horizontal Scaling

### Menjalankan Model Serving
Model dapat dijalankan sebagai REST API menggunakan script serving:
```bash
python src/api/serve.py
```
Endpoint tersedia di `http://localhost:5001`

### Endpoint yang Tersedia
| Endpoint | Method | Fungsi |
|---|---|---|
| `/ping` | GET | Health check |
| `/health` | GET | Status API |
| `/version` | GET | Info model |
| `/invocations` | POST | Prediksi fraud |

### Contoh Request Prediksi
```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "is_sell": 1,
    "amount_scaled": 1.5,
    "volume_scaled": 0.3,
    "hour": 14,
    "minute": 30
  }'
```

### Horizontal Scaling dengan Docker Compose
```bash
# Jalankan 3 replika
docker compose up -d

# Cek status semua replika
docker compose ps

# Scale up ke 5 replika
docker compose up -d --scale api-service=5

# Scale down ke 1 replika
docker compose up -d --scale api-service=1
```

## Menjalankan Sistem dengan Docker Compose

```bash
docker compose up -d
docker compose ps
```

## Alur Versioning Data dengan DVC

### Menambahkan versi data baru
```bash
python src/data/stream_generator.py
dvc add data/raw/streaming/
git add .
git commit -m "data: add new streaming batch"
git push
```

### Cek status dan perbandingan versi
```bash
dvc status
dvc diff
```

## Model Registry & Versioning

| Detail | Value |
|---|---|
| Nama Model | fraud-detection-best-model |
| Versi Aktif | v6 |
| Stage | Production |
| Algoritma | RandomForestClassifier |
| Best F1 Score | 1.0000 |

## Tech Stack

| Komponen | Tools |
|---|---|
| Sumber Data | Kraken Public API (XBTUSD) |
| Data versioning | DVC + DagsHub |
| Experiment tracking | MLflow |
| Model serving | FastAPI + MLflow Registry |
| Drift detection | Evidently AI |
| CI/CD | GitHub Actions |
| Orkestrasi | Docker Compose (3 replicas) |
| ML Framework | scikit-learn |

## Branching Strategy

Proyek ini menggunakan **GitHub Flow**:
- `main` — branch production, hanya menerima merge dari Pull Request
- `feat/*` — branch untuk pengembangan fitur atau eksperimen baru