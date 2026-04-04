import pandas as pd
from pathlib import Path
from datetime import datetime

def load_raw_data(path: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """Load raw dataset dari file CSV."""
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    return df

def ingest_latest_batch(df: pd.DataFrame) -> str:
    """
    Simulasi penarikan data terbaru secara periodik.
    Setiap kali dijalankan, mengambil sampel acak dan menyimpan
    dengan timestamp agar tidak menimpa data lama.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    batch = df.sample(n=10000, random_state=None)
    
    output_path = f"data/raw/batch_{timestamp}.csv"
    batch.to_csv(output_path, index=False)
    print(f"New batch saved: {len(batch)} rows -> {output_path}")
    return output_path

if __name__ == "__main__":
    print("=== Starting Data Ingestion ===")
    df = load_raw_data()
    output_path = ingest_latest_batch(df)
    print(f"=== Ingestion Complete: {output_path} ===")