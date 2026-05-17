import time
import pandas as pd
from binance.client import Client
from datetime import datetime
from pathlib import Path

# Binance public API - tidak perlu API key
client = Client("", "")

def get_crypto_trades(symbol: str = "BTCUSDT", limit: int = 100) -> pd.DataFrame:
    """Ambil data transaksi crypto terbaru dari Binance Public API."""
    trades = client.get_recent_trades(symbol=symbol, limit=limit)
    df = pd.DataFrame(trades)
    df = df[["time", "price", "qty", "isBuyerMaker"]]
    df.columns = ["timestamp", "amount", "quantity", "is_sell"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["amount"]    = df["amount"].astype(float)
    df["quantity"]  = df["quantity"].astype(float)
    df["symbol"]    = symbol
    return df

def label_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """Label transaksi sebagai fraud berdasarkan rules statistik."""
    amount_threshold   = df["amount"].mean() + 2 * df["amount"].std()
    quantity_threshold = df["quantity"].mean() + 2 * df["quantity"].std()
    df["Class"] = 0
    df.loc[df["amount"] > amount_threshold, "Class"]     = 1
    df.loc[df["quantity"] > quantity_threshold, "Class"] = 1
    fraud_count = df["Class"].sum()
    print(f"Fraud detected: {fraud_count}/{len(df)} ({fraud_count/len(df)*100:.2f}%)")
    return df

def ingest_latest_batch(symbol: str = "BTCUSDT") -> str:
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
    print("=== Starting Binance Data Ingestion ===")
    output_path = ingest_latest_batch()
    print(f"=== Ingestion Complete: {output_path} ===")