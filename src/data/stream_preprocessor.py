import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def load_all_batches() -> pd.DataFrame:
    streaming_path = Path("data/raw/streaming")
    files = sorted(streaming_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Tidak ada batch streaming!")
    dfs = [pd.read_csv(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"Total rows from {len(files)} batches: {len(df)}")
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Cleaning — hanya buang missing values dan duplikat PERSIS sama di semua kolom
    before = len(df)
    df = df.dropna()
    df = df.drop_duplicates()  # duplikat persis sama semua kolom, bukan hanya timestamp
    after = len(df)
    print(f"Cleaning: {before - after} rows removed, {after} rows remaining")

    # Fit scaler pada seluruh data
    scaler_amount = StandardScaler()
    scaler_volume = StandardScaler()
    df["amount_scaled"] = scaler_amount.fit_transform(df[["amount"]])
    df["volume_scaled"] = scaler_volume.fit_transform(df[["volume"]])

    # Simpan scaler sebagai artifact
    Path("models/scalers").mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_amount, "models/scalers/scaler_amount.pkl")
    joblib.dump(scaler_volume, "models/scalers/scaler_volume.pkl")
    print("Scalers saved: models/scalers/scaler_amount.pkl & scaler_volume.pkl")

    # Feature engineering
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed')
    df["hour"]      = df["timestamp"].dt.hour
    df["minute"]    = df["timestamp"].dt.minute
    df["is_sell"]   = (df["side"] == "s").astype(int)

    df = df[["amount_scaled", "volume_scaled", "hour", "minute", "is_sell", "Class"]]
    print(f"Features built: {df.shape[1]} columns")
    print(f"Class distribution: {df['Class'].value_counts().to_dict()}")
    return df

def save_processed(df: pd.DataFrame) -> str:
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
