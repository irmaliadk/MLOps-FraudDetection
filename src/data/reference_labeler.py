"""
Reference-based Fraud Labeler
Menggunakan pola statistik dari Kaggle Crypto Scam Dataset
sebagai referensi untuk melabeli data Kraken XBTUSD.

Referensi dataset: muhammadhussnain09/crypto-scam-transaction-dataset
Fraud rate referensi: 7.25%
Pola fraud:
- Amount lebih tinggi dari rata-rata + 0.077 * std (7.7% lebih tinggi)
- Velocity (volume) lebih tinggi dari rata-rata
- Kombinasi keduanya meningkatkan probabilitas fraud
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Konstanta dari analisis Kaggle dataset
KAGGLE_FRAUD_RATE     = 0.0725  # 7.25%
KAGGLE_AMOUNT_RATIO   = 799.98 / 742.52  # fraud/legit amount ratio = 1.077
KAGGLE_VELOCITY_RATIO = 0.0129 / 0.0114  # fraud/legit velocity ratio = 1.132

def compute_fraud_score(df: pd.DataFrame) -> pd.Series:
    """
    Hitung fraud score berdasarkan pola dari Kaggle dataset.
    Score 0-1, semakin tinggi semakin mencurigakan.
    """
    amount_mean  = df["amount"].mean()
    amount_std   = df["amount"].std()
    volume_mean  = df["volume"].mean()
    volume_std   = df["volume"].std()

    # Normalize amount dan volume ke z-score
    amount_z = (df["amount"] - amount_mean) / (amount_std + 1e-10)
    volume_z = (df["volume"] - volume_mean) / (volume_std + 1e-10)

    # Fraud score berdasarkan pola Kaggle:
    # - Amount tinggi lebih berkontribusi (ratio 1.077)
    # - Volume tinggi juga berkontribusi (ratio 1.132)
    fraud_score = (
        0.6 * (amount_z / (KAGGLE_AMOUNT_RATIO * 10)) +
        0.4 * (volume_z / (KAGGLE_VELOCITY_RATIO * 10))
    )

    # Normalize ke 0-1
    fraud_score = (fraud_score - fraud_score.min()) / \
                  (fraud_score.max() - fraud_score.min() + 1e-10)

    return fraud_score

def label_fraud_reference_based(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label fraud berdasarkan referensi pola dari Kaggle dataset.
    Target fraud rate mendekati 7.25% (sesuai referensi).
    """
    fraud_score = compute_fraud_score(df)

    # Tentukan threshold agar fraud rate mendekati referensi Kaggle (7.25%)
    threshold = np.percentile(fraud_score, 100 * (1 - KAGGLE_FRAUD_RATE))

    df = df.copy()
    df["fraud_score"] = fraud_score
    df["Class"]       = (fraud_score >= threshold).astype(int)

    fraud_count = df["Class"].sum()
    fraud_rate  = fraud_count / len(df)

    print(f"Reference fraud rate (Kaggle): {KAGGLE_FRAUD_RATE*100:.2f}%")
    print(f"Actual fraud rate (labeled)  : {fraud_rate*100:.2f}%")
    print(f"Fraud threshold score        : {threshold:.4f}")
    print(f"Fraud cases labeled          : {fraud_count}/{len(df)}")

    return df

if __name__ == "__main__":
    print("=== Reference-based Fraud Labeler ===")
    print(f"Using patterns from Kaggle Crypto Scam Dataset")
    print(f"Amount fraud ratio : {KAGGLE_AMOUNT_RATIO:.3f}x")
    print(f"Velocity fraud ratio: {KAGGLE_VELOCITY_RATIO:.3f}x")
    print()

    # Test dengan data Kraken terbaru
    streaming_path = Path("data/raw/streaming")
    files = sorted([f for f in streaming_path.glob("*.csv")
                   if "shifted" not in f.name])

    if not files:
        print("No streaming data found!")
    else:
        df = pd.read_csv(files[-1])
        print(f"Loaded: {files[-1].name} ({len(df)} rows)")
        df = label_fraud_reference_based(df)
        print(df[["amount", "volume", "fraud_score", "Class"]].describe())