import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

GLOBAL_STATS_PATH = "models/scalers/global_stats.json"

# Konstanta dari analisis Kaggle Crypto Scam Dataset
KAGGLE_FRAUD_RATE     = 0.0725  # 7.25%
KAGGLE_AMOUNT_RATIO   = 799.98 / 742.52   # 1.077x
KAGGLE_VELOCITY_RATIO = 0.0129 / 0.0114   # 1.132x

def get_crypto_trades(symbol: str = "XBTUSD", limit: int = 1000) -> pd.DataFrame:
    """Ambil data transaksi crypto terbaru dari Kraken Public API."""
    url      = "https://api.kraken.com/0/public/Trades"
    params   = {"pair": symbol, "count": limit}
    response = requests.get(url, params=params)
    data     = response.json()

    if data["error"]:
        raise Exception(f"Kraken API error: {data['error']}")

    pair_key = list(data["result"].keys())[0]
    trades   = data["result"][pair_key]

    df = pd.DataFrame(trades, columns=[
        "amount", "volume", "timestamp", "side", "order_type", "misc", "trade_id"
    ])
    df["amount"]    = df["amount"].astype(float)
    df["volume"]    = df["volume"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["symbol"]    = symbol
    df = df[["timestamp", "amount", "volume", "side", "symbol"]]
    return df

def load_global_stats() -> dict:
    """Load statistik global jika sudah ada."""
    if Path(GLOBAL_STATS_PATH).exists():
        with open(GLOBAL_STATS_PATH) as f:
            return json.load(f)
    return None

def update_global_stats(df: pd.DataFrame, existing_stats: dict = None) -> dict:
    """Update statistik global dengan data baru menggunakan weighted average."""
    new_stats = {
        "amount_mean": float(df["amount"].mean()),
        "amount_std":  float(df["amount"].std()),
        "volume_mean": float(df["volume"].mean()),
        "volume_std":  float(df["volume"].std()),
        "n_batches":   1,
        "updated_at":  datetime.now().isoformat()
    }

    if existing_stats:
        n_old = existing_stats.get("n_batches", 1)
        n_new = 1
        total = n_old + n_new
        new_stats["amount_mean"] = (
            existing_stats["amount_mean"] * n_old +
            new_stats["amount_mean"] * n_new
        ) / total
        new_stats["amount_std"]  = max(existing_stats["amount_std"],  new_stats["amount_std"])
        new_stats["volume_mean"] = (
            existing_stats["volume_mean"] * n_old +
            new_stats["volume_mean"] * n_new
        ) / total
        new_stats["volume_std"]  = max(existing_stats["volume_std"],  new_stats["volume_std"])
        new_stats["n_batches"]   = total

    Path(GLOBAL_STATS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_STATS_PATH, "w") as f:
        json.dump(new_stats, f, indent=2)

    return new_stats

def label_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label fraud berdasarkan referensi pola dari Kaggle Crypto Scam Dataset.
    Menggunakan fraud score berbasis z-score dengan bobot dari pola Kaggle.
    Target fraud rate: 7.25% (sesuai referensi dataset nyata).
    """
    amount_mean = df["amount"].mean()
    amount_std  = df["amount"].std()
    volume_mean = df["volume"].mean()
    volume_std  = df["volume"].std()

    amount_z = (df["amount"] - amount_mean) / (amount_std + 1e-10)
    volume_z = (df["volume"] - volume_mean) / (volume_std + 1e-10)

    fraud_score = (
        0.6 * (amount_z / (KAGGLE_AMOUNT_RATIO * 10)) +
        0.4 * (volume_z / (KAGGLE_VELOCITY_RATIO * 10))
    )

    fraud_score = (fraud_score - fraud_score.min()) / \
                  (fraud_score.max() - fraud_score.min() + 1e-10)

    threshold   = float(np.percentile(fraud_score, 100 * (1 - KAGGLE_FRAUD_RATE)))
    df          = df.copy()
    df["Class"] = (fraud_score >= threshold).astype(int)

    fraud_count = df["Class"].sum()
    fraud_rate  = fraud_count / len(df)

    print(f"  Fraud rate (Kaggle ref): {KAGGLE_FRAUD_RATE*100:.2f}%")
    print(f"  Fraud rate (labeled)  : {fraud_rate*100:.2f}%")
    print(f"  Fraud cases           : {fraud_count}/{len(df)}")

    return df

def save_batch(df: pd.DataFrame, symbol: str = "XBTUSD") -> str:
    """Simpan batch dengan timestamp agar tidak menimpa data lama."""
    Path("data/raw/streaming").mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/raw/streaming/{symbol}_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    return output_path

def run_stream(symbol: str = "XBTUSD", interval_seconds: int = 30, max_batches: int = 3):
    """Jalankan streaming data secara berkala."""
    print(f"Starting stream for {symbol}...")
    print(f"Fetching every {interval_seconds}s, max {max_batches} batches")
    print("=" * 50)

    for batch_num in range(1, max_batches + 1):
        print(f"\nBatch {batch_num}/{max_batches} — {datetime.now().strftime('%H:%M:%S')}")

        df             = get_crypto_trades(symbol=symbol)
        existing_stats = load_global_stats()
        global_stats   = update_global_stats(df, existing_stats)
        print(f"  Global stats updated (n_batches={global_stats['n_batches']})")

        df = label_fraud(df)
        save_batch(df, symbol=symbol)

        if batch_num < max_batches:
            print(f"  Waiting {interval_seconds}s...")
            time.sleep(interval_seconds)

    print("\nStreaming complete!")

if __name__ == "__main__":
    run_stream(symbol="XBTUSD", interval_seconds=30, max_batches=5)