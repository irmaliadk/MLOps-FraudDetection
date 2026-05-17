import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

def get_crypto_trades(symbol: str = "XBTUSD", limit: int = 100) -> pd.DataFrame:
    """Ambil data transaksi crypto terbaru dari Kraken Public API."""
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

def label_fraud(df: pd.DataFrame, amount_threshold: float = None, volume_threshold: float = None) -> pd.DataFrame:
    """Label transaksi sebagai fraud berdasarkan rules statistik."""
    if amount_threshold is None:
        amount_threshold = df["amount"].mean() + 2 * df["amount"].std()
    if volume_threshold is None:
        volume_threshold = df["volume"].mean() + 2 * df["volume"].std()
    df["Class"] = 0
    df.loc[df["amount"] > amount_threshold, "Class"] = 1
    df.loc[df["volume"] > volume_threshold, "Class"] = 1
    fraud_count = df["Class"].sum()
    print(f"Fraud detected: {fraud_count}/{len(df)} ({fraud_count/len(df)*100:.2f}%)")
    return df

def ingest_latest_batch(symbol: str = "XBTUSD") -> str:
    """Ambil batch terbaru dan simpan dengan timestamp."""
    Path("data/raw/streaming").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = get_crypto_trades(symbol=symbol)
    df = label_fraud(df)
    output_path = f"data/raw/streaming/{symbol}_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"New batch saved: {len(df)} rows -> {output_path}")
    return output_path

if __name__ == "__main__":
    print("=== Starting Kraken Data Ingestion ===")
    output_path = ingest_latest_batch()
    print(f"=== Ingestion Complete: {output_path} ===")