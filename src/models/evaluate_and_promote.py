"""
Script evaluasi komparatif otomatis.
Membandingkan model baru dengan model lama di registry.
Model baru hanya dipromosikan ke Production jika F1 lebih baik.
"""
import mlflow
import joblib
import pandas as pd
import json
from pathlib import Path
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def get_current_champion_f1() -> float:
    """Ambil F1 Score model champion saat ini dari MLflow."""
    try:
        client = MlflowClient()
        alias_mv = client.get_model_version_by_alias(
            "fraud-detection-best-model", "champion"
        )
        run = client.get_run(alias_mv.run_id)
        f1 = run.data.metrics.get("f1_score", 0.0)
        print(f"Current champion F1: {f1:.4f} (version {alias_mv.version})")
        return f1, alias_mv.version
    except Exception as e:
        print(f"No champion found: {e}")
        return 0.0, None

def evaluate_new_model() -> float:
    """Evaluasi model baru yang baru saja ditraining."""
    model = joblib.load("models/trained/fraud_model.pkl")

    streaming_path = Path("data/processed/streaming")
    files = sorted(streaming_path.glob("*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True).drop_duplicates()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    print(f"New model F1    : {f1:.4f}")
    return f1

def promote_if_better():
    """Promosikan model baru ke champion jika lebih baik."""
    current_f1, current_version = get_current_champion_f1()
    new_f1 = evaluate_new_model()

    comparison = {
        "current_champion_version": current_version,
        "current_champion_f1":      round(current_f1, 4),
        "new_model_f1":             round(new_f1, 4),
        "improvement":              round(new_f1 - current_f1, 4),
        "promoted":                 False
    }

    if new_f1 >= current_f1:
        print(f"\n✅ New model is better or equal ({new_f1:.4f} >= {current_f1:.4f})")
        print("Registering and promoting new model...")

        client   = MlflowClient()
        experiment = client.get_experiment_by_name("fraud-detection-experiments")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1_score DESC"],
            max_results=1
        )

        if runs:
            best_run   = runs[0]
            model_uri  = f"runs:/{best_run.info.run_id}/model"
            result     = mlflow.register_model(
                model_uri=model_uri,
                name="fraud-detection-best-model"
            )
            client.set_registered_model_alias(
                name="fraud-detection-best-model",
                alias="champion",
                version=result.version
            )
            print(f"Model v{result.version} promoted to champion!")
            comparison["promoted"]        = True
            comparison["new_version"]     = result.version
    else:
        print(f"\n⚠️  New model is worse ({new_f1:.4f} < {current_f1:.4f})")
        print("Keeping current champion. No promotion.")

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved: reports/model_comparison.json")

    return comparison

if __name__ == "__main__":
    result = promote_if_better()
    print(f"\nResult: {result}")