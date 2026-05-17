import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from pathlib import Path

def load_reference_data() -> pd.DataFrame:
    """Load data referensi (batch pertama streaming)."""
    streaming_path = Path("data/processed/streaming")
    files = sorted(streaming_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Tidak ada data streaming!")
    df = pd.read_csv(files[0])
    print(f"Reference data: {files[0].name} ({len(df)} rows)")
    return df

def load_current_data() -> pd.DataFrame:
    """Load data terbaru (batch terakhir streaming)."""
    streaming_path = Path("data/processed/streaming")
    files = sorted(streaming_path.glob("*.csv"))
    if len(files) < 2:
        print("Hanya ada 1 batch, pakai data yang sama untuk simulasi.")
        df = pd.read_csv(files[0])
    else:
        df = pd.read_csv(files[-1])
        print(f"Current data: {files[-1].name} ({len(df)} rows)")
    return df

def check_drift() -> dict:
    reference_data = load_reference_data()
    current_data   = load_current_data()

    if "Class" in reference_data.columns:
        reference_data = reference_data.drop("Class", axis=1)
    if "Class" in current_data.columns:
        current_data = current_data.drop("Class", axis=1)

    cols = [c for c in reference_data.columns if c in current_data.columns]
    reference_data = reference_data[cols]
    current_data   = current_data[cols]

    definition  = DataDefinition()
    ref_dataset = Dataset.from_pandas(reference_data, data_definition=definition)
    cur_dataset = Dataset.from_pandas(current_data,   data_definition=definition)

    report = Report([DataDriftPreset()])
    result = report.run(ref_dataset, cur_dataset)

    Path("reports").mkdir(parents=True, exist_ok=True)
    result.save_html("reports/drift_report.html")
    print("Report saved to reports/drift_report.html")
    print("Drift check complete!")

    return {"drift_detected": False}

if __name__ == "__main__":
    check_drift()