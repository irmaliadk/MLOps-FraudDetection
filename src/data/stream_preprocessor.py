import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def load_latest_batch() -> pd.DataFrame:
    """Load batch terbaru dari folder streaming."""
    streaming_path = Path("data/raw/streaming")
    files = sorted(streaming_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Tidak ada batch streaming!")
    latest_file = files[-1]
    print(f"Loading: {latest_file}")
    return pd.read_csv(latest_file)

def load_all_batches() -> pd.DataFrame:
    """Gabungkan semua batch streaming menjadi satu dataset."""
    streaming_path = Path("data/raw/streaming")
    files = sorted(streaming_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Tidak ada batch streaming!")
    dfs = [pd.read_csv(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"Total rows from {len(files)} batches: {len(df)}")
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning dan feature engineering untuk data streaming Kraken."""
    before = len(df)
    df = df.dropna()
    df = df.drop_duplicates(subset=["timestamp"])
    after = len(df)
    print(f"Cleaning: {before - after} rows removed, {after} rows remaining")

    scaler = StandardScaler()
    df["amount_scaled"] = scaler.fit_transform(df[["amount"]])
    df["volume_scaled"] = scaler.fit_transform(df[["volume"]])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]      = df["timestamp"].dt.hour
    df["minute"]    = df["timestamp"].dt.minute
    df["is_sell"]   = (df["side"] == "s").astype(int)

    df = df.drop(["timestamp", "amount", "volume", "side", "symbol"], axis=1)
    print(f"Features built: {df.shape[1]} columns")
    return df

def save_processed(df: pd.DataFrame) -> str:
    """Simpan data processed dengan timestamp."""
    Path("data/processed/streaming").mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/processed/streaming/processed_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    print("=== Starting Stream Preprocessing ===")
    df = load_all_batches()
    df = preprocess(df)
    save_processed(df)
    print("=== Preprocessing Complete ===")