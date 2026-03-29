# MLOps Fraud Detection System

Sistem deteksi fraud pada transaksi perbankan yang dibangun dengan pendekatan MLOps,
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

## Dataset

Menggunakan [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dari Kaggle — 284.807 transaksi kartu kredit, dengan 492 kasus fraud (0.17%).

## Struktur Direktori
```
MLOps-FraudDetection/
├── .devcontainer/         # Konfigurasi GitHub Codespaces
│   └── devcontainer.json
├── data/
│   ├── raw/               # Data mentah dari sumber
│   ├── processed/         # Data setelah cleaning & feature engineering
│   └── external/          # Data referensi eksternal
├── models/
│   ├── trained/           # Model hasil training
│   └── registry/          # Model registry (versioning)
├── notebooks/             # Jupyter notebooks untuk eksplorasi
├── src/
│   ├── data/              # Script ingestion & preprocessing
│   ├── features/          # Script feature engineering
│   ├── models/            # Script training & evaluasi
│   ├── api/               # FastAPI inference endpoint
│   └── monitoring/        # Drift detection & monitoring
├── config/                # File konfigurasi
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md
```

## Cara Menjalankan dengan GitHub Codespaces

1. Buka repositori ini di GitHub
2. Klik tombol hijau **"Code"**
3. Pilih tab **"Codespaces"**
4. Klik **"Create codespace on main"**
5. Tunggu environment selesai dibangun (otomatis install semua dependencies)
6. Selesai — environment siap digunakan tanpa setup manual

## Tech Stack

| Komponen | Tools |
|---|---|
| Data versioning | DVC |
| Experiment tracking | MLflow |
| Model serving | FastAPI |
| Drift detection | Evidently AI |
| CI/CD | GitHub Actions |
| ML Framework | scikit-learn |

## Branching Strategy

Proyek ini menggunakan **GitHub Flow**:
- `main` — branch production, hanya menerima merge dari Pull Request
- `feat/*` — branch untuk pengembangan fitur atau eksperimen baru