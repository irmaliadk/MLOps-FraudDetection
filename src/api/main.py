import joblib
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="Fraud Detection API",
    version="3.0.0",
    description="Inference API untuk deteksi fraud transaksi crypto XBTUSD"
)

# Load model dari MLflow Registry
try:
    model = mlflow.pyfunc.load_model("models:/fraud-detection-best-model@champion")
    print("Model loaded from MLflow Registry (champion)")
except Exception:
    model = joblib.load("models/trained/fraud_model.pkl")
    print("Model loaded from local pkl (fallback)")

# Load scaler yang sama dengan yang dipakai saat training
scaler_amount = joblib.load("models/scalers/scaler_amount.pkl")
scaler_volume = joblib.load("models/scalers/scaler_volume.pkl")
print("Scalers loaded successfully")

# User kirim nilai RAW dari Kraken, bukan pre-scaled
class Transaction(BaseModel):
    is_sell: int        # 1 = sell, 0 = buy
    amount: float       # harga asli, contoh: 103500.25
    volume: float       # volume asli, contoh: 0.00234
    hour: int           # jam transaksi (0-23)
    minute: int         # menit transaksi (0-59)

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
    return {"message": "Fraud Detection API is running!", "version": "3.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Transform raw input pakai scaler yang SAMA dengan training
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
            "scaled_values":     {
                "amount_scaled": round(amount_scaled, 4),
                "volume_scaled": round(volume_scaled, 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
