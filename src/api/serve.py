"""
MLflow Model Serving Script
Endpoint /invocations mengikuti konvensi mlflow models serve.
"""
import joblib
import mlflow.pyfunc
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import os

DAGSHUB_URI = "https://dagshub.com/irmaliadk/MLOps-FraudDetection.mlflow"
LOCAL_URI = "sqlite:///mlflow.db"

if os.getenv("DAGSHUB_TOKEN"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = "irmaliadk"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(DAGSHUB_URI)
    mlflow.set_registry_uri(DAGSHUB_URI)
else:
    mlflow.set_tracking_uri(LOCAL_URI)
    mlflow.set_registry_uri(LOCAL_URI)

app = FastAPI(
    title="MLflow Model Serving",
    description="Serving fraud-detection-best-model from MLflow Registry",
    version="1.0.0"
)

model = mlflow.pyfunc.load_model("models:/fraud-detection-best-model@champion")
scaler_amount = joblib.load("models/scalers/scaler_amount.pkl")
scaler_volume = joblib.load("models/scalers/scaler_volume.pkl")
print("Model and scalers loaded successfully!")

class Transaction(BaseModel):
    is_sell: int
    amount: float       # nilai raw dari Kraken
    volume: float       # nilai raw dari Kraken
    hour: int
    minute: int

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/version")
def version():
    return {
        "model_name": "fraud-detection-best-model",
        "alias":      "champion",
        "serving":    "MLflow Models via FastAPI"
    }

@app.post("/invocations")
def invocations(transaction: Transaction):
    """Endpoint prediksi — konvensi mlflow models serve."""
    amount_scaled = float(scaler_amount.transform([[transaction.amount]])[0][0])
    volume_scaled = float(scaler_volume.transform([[transaction.volume]])[0][0])

    input_df = pd.DataFrame([{
        "amount_scaled": amount_scaled,
        "volume_scaled": volume_scaled,
        "hour":          transaction.hour,
        "minute":        transaction.minute,
        "is_sell":       transaction.is_sell,
    }])

    prediction = int(model.predict(input_df)[0])
    return {
        "predictions": [prediction],
        "label":       "FRAUD" if prediction == 1 else "LEGITIMATE"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
