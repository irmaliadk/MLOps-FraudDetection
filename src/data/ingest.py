import pandas as pd
from pathlib import Path

def load_raw_data(path: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """Load raw dataset dari file CSV."""
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    return df

def simulate_weekly_batch(week_number: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulasi data streaming mingguan.
    Dataset dibagi jadi 4 batch, masing-masing mewakili 1 minggu.
    """
    chunk_size = len(df) // 4
    start = week_number * chunk_size
    end = (week_number + 1) * chunk_size
    batch = df.iloc[start:end].copy()

    output_path = f"data/raw/week_{week_number}.csv"
    batch.to_csv(output_path, index=False)
    print(f"Week {week_number} batch saved: {len(batch)} rows -> {output_path}")
    return batch

if __name__ == "__main__":
    df = load_raw_data()
    for week in range(4):
        simulate_weekly_batch(week, df)