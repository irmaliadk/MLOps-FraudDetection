"""
Script simulasi data drift.
Membuat data yang distribusinya berbeda dari data training normal
untuk menguji kemampuan drift detector.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def simulate_shifted_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Buat data dengan distribusi yang sengaja di-shift
    untuk mensimulasikan data drift.
    """
    np.random.seed(42)

    # Normal data: BTC price sekitar 78000-79000
    # Shifted data: BTC price naik drastis ke 95000-100000 (market crash scenario)
    amount = np.random.normal(loc=97000, scale=500, size=n_samples)
    volume = np.random.normal(loc=2.5, scale=0.5, size=n_samples)  # volume jauh lebih tinggi
    hour   = np.random.randint(0, 24, size=n_samples)
    minute = np.random.randint(0, 60, size=n_samples)
    side   = np.random.choice(["b", "s"], size=n_samples)

    df = pd.DataFrame({
        "amount":    amount,
        "volume":    volume,
        "timestamp": pd.date_range(start="2026-06-01", periods=n_samples, freq="s"),
        "side":      side,
        "symbol":    "XBTUSD"
    })

    # Label fraud berdasarkan threshold yang sama
    amount_threshold = df["amount"].mean() + df["amount"].std()
    volume_threshold = df["volume"].mean() + df["volume"].std()
    df["Class"] = 0
    df.loc[df["amount"] > amount_threshold, "Class"] = 1
    df.loc[df["volume"] > volume_threshold, "Class"] = 1

    return df

def save_shifted_data(df: pd.DataFrame):
    """Simpan shifted data ke folder raw streaming."""
    Path("data/raw/streaming").mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/raw/streaming/XBTUSD_shifted_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"Shifted data saved: {output_path} ({len(df)} rows)")
    return output_path

if __name__ == "__main__":
    print("=== Simulating Data Drift ===")
    print("Creating shifted data with different distribution...")
    print("Normal BTC price: ~78000-79000")
    print("Shifted BTC price: ~95000-100000 (market crash scenario)")
    print()

    df = simulate_shifted_data(n_samples=1000)
    output_path = save_shifted_data(df)

    print(f"\nData statistics:")
    print(f"Amount mean: {df['amount'].mean():.2f} (normal: ~78500)")
    print(f"Volume mean: {df['volume'].mean():.2f} (normal: ~0.05)")
    print(f"Fraud rate : {df['Class'].mean()*100:.1f}%")
    print("\nNow run stream_preprocessor.py and drift_detector.py to check drift!")