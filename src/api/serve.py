"""
MLflow Model Serving Script
Mensimulasikan fungsi 'mlflow models serve' menggunakan FastAPI.
Model dimuat langsung dari MLflow Production Registry.
"""
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="MLflow Model Serving",
    description="Serving fraud-detection-best-model from MLflow Production Registry",
    version="1.0.0"
)

print("Loading model from MLflow Production Registry...")
model = mlflow.pyfunc.load_model("models:/fraud-detection-best-model/Production")
print("Model loaded successfully!")

class Transaction(BaseModel):
    is_sell: int
    amount_scaled: float
    volume_scaled: float
    hour: int
    minute: int

@app.get("/ping")
def ping():
    """Health check endpoint — sama seperti mlflow models serve."""
    return {"status": "alive"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/version")
def version():
    return {
        "model_name": "fraud-detection-best-model",
        "stage": "Production",
        "serving": "MLflow Models via FastAPI"
    }

@app.post("/invocations")
def invocations(transaction: Transaction):
    """
    Endpoint prediksi — sama seperti mlflow models serve /invocations.
    """
    input_df = pd.DataFrame([{
        "amount_scaled": transaction.amount_scaled,
        "volume_scaled": transaction.volume_scaled,
        "hour":          transaction.hour,
        "minute":        transaction.minute,
        "is_sell":       transaction.is_sell,
    }])
    prediction = model.predict(input_df)[0]
    return {
        "predictions": [int(prediction)],
        "label": "FRAUD" if prediction == 1 else "LEGITIMATE"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)