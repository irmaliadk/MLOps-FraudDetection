import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(title="Fraud Detection API", version="2.0.0")

model = mlflow.pyfunc.load_model("models:/fraud-detection-best-model/Production")
print("Model loaded from MLflow Production registry!")

class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!", "data_source": "Binance BTCUSDT"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction):
    import pandas as pd
    features = np.array(transaction.features).reshape(1, -1)
    df = pd.DataFrame(features)
    prediction = model.predict(df)[0]
    return {
        "prediction": int(prediction),
        "label": "FRAUD" if prediction == 1 else "LEGITIMATE"
    }