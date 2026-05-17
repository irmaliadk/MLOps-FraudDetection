import time
import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

GLOBAL_STATS_PATH = "models/scalers/global_stats.json"

def get_crypto_trades(symbol: str = "XBTUSD", limit: int = 100) -> pd.DataFrame:
    """Ambil data transaksi crypto terbaru dari Kraken Public API."""
    url     = "https://api.kraken.com/0/public/Trades"
    params  = {"pair": symbol, "count": limit}
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
    """
    Load statistik global jika sudah ada.
    Statistik ini dipakai sebagai threshold fraud yang KONSISTEN
    antar batch — bukan dihitung ulang per batch.
    """
    if Path(GLOBAL_STATS_PATH).exists():
        with open(GLOBAL_STATS_PATH) as f:
            return json.load(f)
    return None

def update_global_stats(df: pd.DataFrame, existing_stats: dict = None) -> dict:
    """
    Update statistik global dengan data baru menggunakan
    running statistics — tanpa menyimpan seluruh data lama.
    """
    new_stats = {
        "amount_mean": float(df["amount"].mean()),
        "amount_std":  float(df["amount"].std()),
        "volume_mean": float(df["volume"].mean()),
        "volume_std":  float(df["volume"].std()),
        "n_batches":   1,
        "updated_at":  datetime.now().isoformat()
    }

    # Jika sudah ada statistik sebelumnya, gabungkan (weighted average)
    if existing_stats:
        n_old = existing_stats.get("n_batches", 1)
        n_new = 1
        total = n_old + n_new
        new_stats["amount_mean"] = (
            existing_stats["amount_mean"] * n_old +
            new_stats["amount_mean"] * n_new
        ) / total
        new_stats["amount_std"] = max(
            existing_stats["amount_std"],
            new_stats["amount_std"]
        )
        new_stats["volume_mean"] = (
            existing_stats["volume_mean"] * n_old +
            new_stats["volume_mean"] * n_new
        ) / total
        new_stats["volume_std"] = max(
            existing_stats["volume_std"],
            new_stats["volume_std"]
        )
        new_stats["n_batches"] = total

    # Simpan ke file
    Path(GLOBAL_STATS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_STATS_PATH, "w") as f:
        json.dump(new_stats, f, indent=2)

    return new_stats

def label_fraud(df: pd.DataFrame, global_stats: dict) -> pd.DataFrame:
    """
    Label fraud berdasarkan statistik GLOBAL yang konsisten.
    Threshold tidak berubah antar batch.
    """
    amount_threshold = global_stats["amount_mean"] + 1 * global_stats["amount_std"]
    volume_threshold = global_stats["volume_mean"] + 1 * global_stats["volume_std"]

    df["Class"] = 0
    df.loc[df["amount"] > amount_threshold, "Class"] = 1
    df.loc[df["volume"] > volume_threshold, "Class"] = 1

    fraud_count = df["Class"].sum()
    print(f"  Fraud threshold amount : {amount_threshold:.2f}")
    print(f"  Fraud threshold volume : {volume_threshold:.6f}")
    print(f"  Fraud detected         : {fraud_count}/{len(df)} ({fraud_count/len(df)*100:.1f}%)")
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

        df = get_crypto_trades(symbol=symbol)

        # Load statistik global lalu update dengan data baru
        existing_stats = load_global_stats()
        global_stats   = update_global_stats(df, existing_stats)
        print(f"  Global stats updated (n_batches={global_stats['n_batches']})")

        # Label fraud pakai threshold global yang konsisten
        df = label_fraud(df, global_stats)
        save_batch(df, symbol=symbol)

        if batch_num < max_batches:
            print(f"  Waiting {interval_seconds}s...")
            time.sleep(interval_seconds)

    print("\nStreaming complete!")

if __name__ == "__main__":
    run_stream(symbol="XBTUSD", interval_seconds=30, max_batches=3)
