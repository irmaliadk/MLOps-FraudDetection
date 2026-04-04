import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus data duplikat dan missing values."""
    before = len(df)
    df = df.dropna()
    df = df.drop_duplicates()
    after = len(df)
    print(f"Cleaning: {before - after} rows removed, {after} rows remaining")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi Amount dan ekstrak fitur Hour dari Time."""
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Hour'] = df['Time'].apply(lambda x: int(x / 3600) % 24)
    df = df.drop(['Time', 'Amount'], axis=1)
    print(f"Features built: {df.shape[1]} columns")
    return df

def save_processed_data(df: pd.DataFrame) -> str:
    """Simpan data processed dengan timestamp."""
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/processed/processed_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    print("=== Starting Preprocessing ===")
    df = pd.read_csv("data/raw/creditcard.csv")
    df = clean_data(df)
    df = engineer_features(df)
    output_path = save_processed_data(df)
    print(f"=== Preprocessing Complete: {output_path} ===")