import time
import pandas as pd
from binance.client import Client
from datetime import datetime
from pathlib import Path

# Binance public API - tidak perlu API key untuk data publik
client = Client("", "")

def get_crypto_trades(symbol: str = "BTCUSDT", limit: int = 100) -> pd.DataFrame:
    """
    Ambil data transaksi crypto terbaru dari Binance.
    Data ini real dan terus bergerak setiap detik.
    """
    trades = client.get_recent_trades(symbol=symbol, limit=limit)
    
    df = pd.DataFrame(trades)
    df = df[["time", "price", "qty", "isBuyerMaker"]]
    df.columns = ["timestamp", "amount", "quantity", "is_sell"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["amount"] = df["amount"].astype(float)
    df["quantity"] = df["quantity"].astype(float)
    df["symbol"] = symbol
    return df

def label_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label transaksi sebagai fraud berdasarkan rules:
    - Amount sangat tinggi (> 2 std dev dari mean) = fraud
    - Quantity sangat tinggi (> 2 std dev dari mean) = fraud
    """
    amount_threshold  = df["amount"].mean() + 2 * df["amount"].std()
    quantity_threshold = df["quantity"].mean() + 2 * df["quantity"].std()
    
    df["Class"] = 0
    df.loc[df["amount"] > amount_threshold, "Class"] = 1
    df.loc[df["quantity"] > quantity_threshold, "Class"] = 1
    
    fraud_count = df["Class"].sum()
    print(f"Fraud detected: {fraud_count}/{len(df)} transactions ({fraud_count/len(df)*100:.2f}%)")
    return df

def save_batch(df: pd.DataFrame, symbol: str = "BTCUSDT") -> str:
    """Simpan batch dengan timestamp agar tidak menimpa data lama."""
    Path("data/raw/streaming").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/raw/streaming/{symbol}_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return output_path

def run_stream(symbol: str = "BTCUSDT", interval_seconds: int = 30, max_batches: int = 5):
    """
    Jalankan streaming data secara berkala.
    Setiap interval_seconds detik, ambil data baru dan simpan.
    """
    print(f"Starting stream for {symbol}...")
    print(f"Fetching every {interval_seconds} seconds, max {max_batches} batches")
    print("="*50)
    
    for batch_num in range(1, max_batches + 1):
        print(f"\nBatch {batch_num}/{max_batches} - {datetime.now().strftime('%H:%M:%S')}")
        df = get_crypto_trades(symbol=symbol)
        df = label_fraud(df)
        save_batch(df, symbol=symbol)
        
        if batch_num < max_batches:
            print(f"Waiting {interval_seconds} seconds for next batch...")
            time.sleep(interval_seconds)
    
    print("\nStreaming complete!")

if __name__ == "__main__":
    run_stream(symbol="BTCUSDT", interval_seconds=30, max_batches=5)