import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy import stats

def load_reference_data() -> pd.DataFrame:
    """Load data referensi dari raw streaming."""
    streaming_path = Path("data/raw/streaming")
    files = sorted([f for f in streaming_path.glob("*.csv")
                   if "shifted" not in f.name])
    if not files:
        raise FileNotFoundError("Tidak ada data streaming!")
    df = pd.read_csv(files[0])
    df = df[["amount", "volume"]].dropna()
    print(f"Reference data: {files[0].name} ({len(df)} rows)")
    return df

def load_current_data() -> pd.DataFrame:
    """Load data terbaru dari raw streaming."""
    streaming_path = Path("data/raw/streaming")
    files = sorted(streaming_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Tidak ada data streaming!")
    df = pd.read_csv(files[-1])
    df = df[["amount", "volume"]].dropna()
    print(f"Current data: {files[-1].name} ({len(df)} rows)")
    return df

def check_drift(drift_threshold: float = 0.3) -> dict:
    """
    Cek data drift menggunakan Kolmogorov-Smirnov test.
    KS test membandingkan distribusi dua dataset secara statistik.
    p-value < 0.05 berarti distribusi berbeda signifikan (drift).
    """
    reference_data = load_reference_data()
    current_data   = load_current_data()

    cols = ["amount", "volume"]
    drifted_cols = 0
    drift_details = {}

    for col in cols:
        ref_vals = reference_data[col].dropna().values
        cur_vals = current_data[col].dropna().values

        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)

        ref_mean = ref_vals.mean()
        cur_mean = cur_vals.mean()
        mean_shift = abs(cur_mean - ref_mean) / (ref_mean + 1e-10)

        col_drift = bool(p_value < 0.05 or mean_shift > 0.1)

        if col_drift:
            drifted_cols += 1

        drift_details[col] = {
            "ks_statistic":  round(float(ks_stat), 4),
            "p_value":       round(float(p_value), 6),
            "ref_mean":      round(float(ref_mean), 4),
            "current_mean":  round(float(cur_mean), 4),
            "mean_shift_pct": round(float(mean_shift * 100), 2),
            "drift_detected": bool(col_drift)
        }

        print(f"  {col}: KS={ks_stat:.4f}, p={p_value:.6f}, "
              f"mean shift={mean_shift*100:.1f}% → "
              f"{'DRIFT' if col_drift else 'OK'}")

    total_cols    = len(cols)
    drift_share   = drifted_cols / total_cols
    drift_detected = drift_share >= drift_threshold

    result_summary = {
        "drift_detected":  drift_detected,
        "drift_share":     round(drift_share, 4),
        "drifted_columns": drifted_cols,
        "total_columns":   total_cols,
        "threshold":       drift_threshold,
        "drift_details":   drift_details,
        "timestamp":       datetime.now().isoformat()
    }

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/drift_result.json", "w") as f:
        json.dump(result_summary, f, indent=2)

    print(f"\nDrift detected : {drift_detected}")
    print(f"Drifted columns: {drifted_cols}/{total_cols} ({drift_share*100:.1f}%)")
    print(f"Threshold      : {drift_threshold*100:.0f}%")
    print(f"Result saved   : reports/drift_result.json")

    return result_summary

if __name__ == "__main__":
    print("=== Checking Data Drift (KS Test) ===")
    result = check_drift(drift_threshold=0.3)
    if result["drift_detected"]:
        print("\n⚠️  DRIFT DETECTED — model perlu diretrain!")
        exit(1)
    else:
        print("\n✅  No drift — model masih relevan.")
        exit(0)