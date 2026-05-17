import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Fraud Detection API",
    version="2.0.0",
    description="Inference API untuk deteksi fraud transaksi crypto BTCUSDT"
)

try:
    model = mlflow.pyfunc.load_model("models:/fraud-detection-best-model@champion")
except Exception:
    import joblib
    model = joblib.load("models/trained/fraud_model.pkl")

class Transaction(BaseModel):
    is_sell: int
    amount_scaled: float
    quantity_scaled: float
    hour: int
    minute: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "is_sell": 0,
                    "amount_scaled": -0.65,
                    "quantity_scaled": -0.29,
                    "hour": 14,
                    "minute": 30
                }
            ]
        }
    }

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        input_df = pd.DataFrame([{
            "is_sell":         transaction.is_sell,
            "amount_scaled":   transaction.amount_scaled,
            "quantity_scaled": transaction.quantity_scaled,
            "hour":            transaction.hour,
            "minute":          transaction.minute,
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
            "input_received":    transaction.model_dump()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
