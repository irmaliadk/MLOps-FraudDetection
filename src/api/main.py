import os
import joblib
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="Fraud Detection API",
    version="3.0.0",
    description="Inference API untuk deteksi fraud transaksi crypto XBTUSD"
)

Instrumentator().instrument(app).expose(app)

# Environment variable untuk kontrol mode loading
USE_MLFLOW_REGISTRY = os.getenv("USE_MLFLOW_REGISTRY", "false").lower() == "true"

if USE_MLFLOW_REGISTRY:
    try:
        model = mlflow.pyfunc.load_model("models:/fraud-detection-best-model@champion")
        print("Model loaded from MLflow Registry (@champion)")
    except Exception as e:
        print(f"MLflow Registry failed: {e}")
        print("Falling back to local pkl...")
        model = joblib.load("models/trained/fraud_model.pkl")
        print("Model loaded from local pkl (fallback)")
else:
    model = joblib.load("models/trained/fraud_model.pkl")
    print("Model loaded from local pkl (USE_MLFLOW_REGISTRY=false)")

# Load scaler
scaler_amount = joblib.load("models/scalers/scaler_amount.pkl")
scaler_volume = joblib.load("models/scalers/scaler_volume.pkl")
print("Scalers loaded successfully")

class Transaction(BaseModel):
    is_sell: int
    amount: float
    volume: float
    hour: int
    minute: int

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "is_sell": 1,
                "amount": 103500.25,
                "volume": 0.00234,
                "hour": 14,
                "minute": 30
            }]
        }
    }

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running!",
        "version": "3.0.0",
        "model_source": "MLflow Registry (@champion)" if USE_MLFLOW_REGISTRY else "Local pkl"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
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

        try:
            sk_model    = model._model_impl.sklearn_model
            fraud_proba = round(float(sk_model.predict_proba(input_df)[0][1]), 4)
        except Exception:
            fraud_proba = float(prediction)

        return {
            "prediction":        prediction,
            "label":             "FRAUD" if prediction == 1 else "LEGITIMATE",
            "fraud_probability": fraud_proba,
            "scaled_values": {
                "amount_scaled": round(amount_scaled, 4),
                "volume_scaled": round(volume_scaled, 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")