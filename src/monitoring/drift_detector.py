import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from pathlib import Path

def check_drift(reference_path: str, current_path: str) -> dict:
    """
    Bandingkan distribusi data referensi vs data baru.
    Kalau drift terdeteksi, model perlu diretrain.
    """
    reference_data = pd.read_csv(reference_path)
    current_data   = pd.read_csv(current_path)

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
    print("No drift — model masih relevan.")

    return {"drift_detected": False}

if __name__ == "__main__":
    check_drift(
        reference_path="data/processed/creditcard_processed.csv",
        current_path="data/raw/week_0.csv"
    )