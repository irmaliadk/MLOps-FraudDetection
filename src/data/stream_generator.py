import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

def get_crypto_trades(symbol: str = "XBTUSD", limit: int = 100) -> pd.DataFrame:
    """
    Ambil data transaksi crypto terbaru dari Kraken Public API.
    Tidak memerlukan API key dan tidak ada restricsi lokasi.
    """
    url    = "https://api.kraken.com/0/public/Trades"
    params = {"pair": symbol, "count": limit}
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

def label_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """Label transaksi sebagai fraud berdasarkan rules statistik."""
    amount_threshold = df["amount"].mean() + 2 * df["amount"].std()
    volume_threshold = df["volume"].mean() + 2 * df["volume"].std()
    df["Class"] = 0
    df.loc[df["amount"] > amount_threshold, "Class"] = 1
    df.loc[df["volume"] > volume_threshold, "Class"] = 1
    fraud_count = df["Class"].sum()
    print(f"Fraud detected: {fraud_count}/{len(df)} ({fraud_count/len(df)*100:.2f}%)")
    return df

def save_batch(df: pd.DataFrame, symbol: str = "XBTUSD") -> str:
    """Simpan batch dengan timestamp agar tidak menimpa data lama."""
    Path("data/raw/streaming").mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/raw/streaming/{symbol}_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return output_path

def run_stream(symbol: str = "XBTUSD", interval_seconds: int = 30, max_batches: int = 3):
    """Jalankan streaming data secara berkala."""
    print(f"Starting stream for {symbol}...")
    print(f"Fetching every {interval_seconds} seconds, max {max_batches} batches")
    print("="*50)

    for batch_num in range(1, max_batches + 1):
        print(f"\nBatch {batch_num}/{max_batches} - {datetime.now().strftime('%H:%M:%S')}")
        df = get_crypto_trades(symbol=symbol)
        df = label_fraud(df)
        save_batch(df, symbol=symbol)

        if batch_num < max_batches:
            print(f"Waiting {interval_seconds} seconds...")
            time.sleep(interval_seconds)

    print("\nStreaming complete!")

if __name__ == "__main__":
    run_stream(symbol="XBTUSD", interval_seconds=30, max_batches=3)