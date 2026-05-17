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

Menggunakan **Binance Public API** — data transaksi crypto BTCUSDT
yang diambil secara real-time setiap 30 detik tanpa memerlukan API key.
Label fraud ditentukan secara otomatis berdasarkan rules statistik:
transaksi dengan amount atau quantity di atas 2 standar deviasi dari
rata-rata diklasifikasikan sebagai fraud.

## Struktur Direktori

```
MLOps-FraudDetection/
├── .devcontainer/         # Konfigurasi GitHub Codespaces
│   └── devcontainer.json
├── .github/
│   └── workflows/
│       ├── retrain.yml            # Weekly retrain otomatis
│       └── mlops-automation.yaml  # End-to-end CI/CD pipeline
├── data/
│   ├── raw/
│   │   └── streaming/     # Data mentah dari Binance API
│   └── processed/
│       └── streaming/     # Data setelah preprocessing
├── models/
│   ├── trained/           # Model hasil training
│   └── registry/          # Model registry
├── notebooks/             # Jupyter notebooks eksplorasi
├── src/
│   ├── data/              # Script ingestion & streaming
│   ├── features/          # Script feature engineering
│   ├── models/            # Script training & evaluasi
│   ├── api/               # FastAPI inference endpoint
│   └── monitoring/        # Drift detection & monitoring
├── config/                # File konfigurasi
├── tests/                 # Unit tests
├── Dockerfile             # Container untuk API service
├── docker-compose.yaml    # Orkestrasi multi-container
├── requirements.txt       # Python dependencies
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

### 1. Fetch data streaming dari Binance
```bash
python src/data/stream_generator.py
```
Output: file baru di `data/raw/streaming/BTCUSDT_YYYYMMDD_HHMMSS.csv`

### 2. Preprocessing data streaming
```bash
python src/data/stream_preprocessor.py
```
Output: file baru di `data/processed/streaming/processed_YYYYMMDD_HHMMSS.csv`

### 3. Training model
```bash
python src/models/train.py
```

### 4. Cek drift
```bash
python src/monitoring/drift_detector.py
```

### 5. Jalankan API inference
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
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
| Versi Aktif | v2 |
| Stage | Production |
| Algoritma | LogisticRegression |
| Best F1 Score | 0.8571 |

## Tech Stack

| Komponen | Tools |
|---|---|
| Sumber Data | Binance Public API |
| Data versioning | DVC |
| Experiment tracking | MLflow |
| Model serving | FastAPI |
| Drift detection | Evidently AI |
| CI/CD | GitHub Actions |
| Orkestrasi | Docker Compose |
| ML Framework | scikit-learn |

## Branching Strategy

Proyek ini menggunakan **GitHub Flow**:
- `main` — branch production, hanya menerima merge dari Pull Request
- `feat/*` — branch untuk pengembangan fitur atau eksperimen baru