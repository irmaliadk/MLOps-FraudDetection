import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from pathlib import Path

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")
artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT", "./mlruns")
os.makedirs(artifact_root, exist_ok=True)
os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_root

def load_data() -> pd.DataFrame:
    """Load data streaming terbaru dari Binance."""
    streaming_path = Path("data/processed/streaming")
    files = sorted(streaming_path.glob("*.csv"))

    if not files:
        raise FileNotFoundError("Tidak ada data streaming! Jalankan stream_generator.py dulu.")

    dfs = [pd.read_csv(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"Using Binance streaming data: {len(df)} rows from {len(files)} batches")
    return df

def run_experiment(model, model_name: str, params: dict, X_train, X_test, y_train, y_test):
    mlflow.set_experiment("fraud-detection-experiments")

    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1        = f1_score(y_test, y_pred)
        auc       = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)

        for key, value in params.items():
            mlflow.log_param(key, value)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Simpan model sebagai MLflow artifact
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"fraud-{model_name}"
        )

        run_id = run.info.run_id

        print(f"\n=== {model_name} ===")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC AUC   : {auc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"Run ID    : {run_id}")

        return f1, model, run_id

if __name__ == "__main__":
    df = load_data()
    X  = df.drop("Class", axis=1)
    y  = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    experiments = [
        {
            "model": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
            "name": "RandomForest_100trees",
            "params": {"model_type": "RandomForest", "n_estimators": 100, "class_weight": "balanced"}
        },
        {
            "model": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"),
            "name": "RandomForest_200trees_depth10",
            "params": {"model_type": "RandomForest", "n_estimators": 200, "max_depth": 10, "class_weight": "balanced"}
        },
        {
            "model": DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced"),
            "name": "DecisionTree_depth10",
            "params": {"model_type": "DecisionTree", "max_depth": 10, "class_weight": "balanced"}
        },
        {
            "model": LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42),
            "name": "LogisticRegression_C0.1",
            "params": {"model_type": "LogisticRegression", "C": 0.1, "max_iter": 1000, "class_weight": "balanced"}
        }
    ]

    best_f1    = 0
    best_model = None
    best_name  = ""
    best_run_id = ""

    for exp in experiments:
        f1, trained_model, run_id = run_experiment(
            model=exp["model"],
            model_name=exp["name"],
            params=exp["params"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        if f1 > best_f1:
            best_f1     = f1
            best_model  = trained_model
            best_name   = exp["name"]
            best_run_id = run_id

    Path("models/trained").mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, "models/trained/fraud_model.pkl")
    print(f"\n✅ Best model: {best_name} with F1 Score: {best_f1:.4f}")
    print(f"Best Run ID: {best_run_id}")
    print("Model saved to models/trained/fraud_model.pkl")