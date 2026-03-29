import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
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

def train_model(X, y):
    """Train model dan log hasilnya ke MLflow."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        f1        = f1_score(y_test, y_pred)
        auc       = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model, "model")

        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC AUC   : {auc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")

        Path("models/trained").mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(model, "models/trained/fraud_model.pkl")
        print("Model saved to models/trained/fraud_model.pkl")

    return model

if __name__ == "__main__":
    X, y = load_processed_data()
    train_model(X, y)