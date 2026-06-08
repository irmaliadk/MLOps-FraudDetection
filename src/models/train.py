import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

DAGSHUB_URI = "https://dagshub.com/irmaliadk/MLOps-FraudDetection.mlflow"
LOCAL_URI   = "sqlite:///mlflow.db"

if os.getenv("DAGSHUB_TOKEN"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = "irmaliadk"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(DAGSHUB_URI)
    mlflow.set_registry_uri(DAGSHUB_URI)
    print("MLflow tracking: DagsHub")
else:
    mlflow.set_tracking_uri(LOCAL_URI)
    mlflow.set_registry_uri(LOCAL_URI)
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT", "./mlruns")
    os.makedirs(artifact_root, exist_ok=True)
    print("MLflow tracking: Local SQLite")

def load_data() -> pd.DataFrame:
    streaming_path = Path("data/processed/streaming")
    files = sorted(streaming_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Tidak ada data! Jalankan preprocessing dulu.")
    dfs = [pd.read_csv(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"Total data: {len(df)} rows dari {len(files)} file")
    print(f"Class distribution: {df['Class'].value_counts().to_dict()}")
    return df

def run_experiment(model, model_name, params, X_train, X_test, y_train, y_test):
    mlflow.set_experiment("fraud-detection-experiments")

    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Hitung metrik dengan zero_division=0 untuk hindari warning dan crash
        f1        = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)

        # ROC AUC hanya valid kalau ada 2 class di test set
        unique_classes = y_test.nunique()
        auc = roc_auc_score(y_test, y_pred) if unique_classes > 1 else 0.5

        for key, value in params.items():
            mlflow.log_param(key, value)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"fraud-{model_name}"
        )

        print(f"\n=== {model_name} ===")
        print(f"F1        : {f1:.4f}")
        print(f"ROC AUC   : {auc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"Run ID    : {run.info.run_id}")

        return f1, model, run.info.run_id

if __name__ == "__main__":
    df = load_data()
    X  = df.drop("Class", axis=1)
    y  = df["Class"]

    # Guard: minimal 20 baris dan ada kedua class
    if len(df) < 20:
        raise ValueError(f"Data terlalu sedikit: {len(df)} baris. Minimum 20.")
    if y.nunique() < 2:
        raise ValueError("Hanya ada satu class di data. Pastikan ada fraud dan non-fraud.")

    # Gunakan stratify hanya kalau kedua class punya cukup sampel
    fraud_count = y.sum()
    use_stratify = fraud_count >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None
    )
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    print(f"Stratify: {'Yes' if use_stratify else 'No (fraud terlalu sedikit)'}\n")

    experiments = [
        {
            "model":  RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
            "name":   "RandomForest_100trees",
            "params": {"model_type": "RandomForest", "n_estimators": 100, "class_weight": "balanced"}
        },
        {
            "model":  RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"),
            "name":   "RandomForest_200trees_depth10",
            "params": {"model_type": "RandomForest", "n_estimators": 200, "max_depth": 10, "class_weight": "balanced"}
        },
        {
            "model":  DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced"),
            "name":   "DecisionTree_depth10",
            "params": {"model_type": "DecisionTree", "max_depth": 10, "class_weight": "balanced"}
        },
        {
            "model":  LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42),
            "name":   "LogisticRegression_C0.1",
            "params": {"model_type": "LogisticRegression", "C": 0.1, "max_iter": 1000, "class_weight": "balanced"}
        }
    ]

    best_f1, best_model, best_name, best_run_id = 0, None, "", ""

    for exp in experiments:
        f1, trained_model, run_id = run_experiment(
            model=exp["model"], model_name=exp["name"],
            params=exp["params"],
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )
        if f1 > best_f1:
            best_f1, best_model = f1, trained_model
            best_name, best_run_id = exp["name"], run_id

    Path("models/trained").mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, "models/trained/fraud_model.pkl")
    print(f"\n✅ Best model: {best_name} | F1: {best_f1:.4f}")
    print(f"Best Run ID: {best_run_id}")
