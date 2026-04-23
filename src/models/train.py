import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from pathlib import Path

def load_processed_data():
    """Load data yang sudah diproses."""
    df = pd.read_csv("data/processed/creditcard_processed.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    print(f"Data loaded: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y

def run_experiment(model, model_name: str, params: dict, X_train, X_test, y_train, y_test):
    """Jalankan satu eksperimen dan log ke MLflow."""
    mlflow.set_experiment("fraud-detection-experiments")

    with mlflow.start_run(run_name=model_name):
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
        mlflow.sklearn.log_model(model, "model")

        print(f"\n=== {model_name} ===")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC AUC   : {auc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")

        return f1

if __name__ == "__main__":
    X, y = load_processed_data()
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

    for exp in experiments:
        f1 = run_experiment(
            model=exp["model"],
            model_name=exp["name"],
            params=exp["params"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        if f1 > best_f1:
            best_f1    = f1
            best_model = exp["model"]
            best_name  = exp["name"]

    Path("models/trained").mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, "models/trained/fraud_model.pkl")
    print(f"\n✅ Best model: {best_name} with F1 Score: {best_f1:.4f}")
    print("Model saved to models/trained/fraud_model.pkl")