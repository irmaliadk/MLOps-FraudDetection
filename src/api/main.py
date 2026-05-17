import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(title="Fraud Detection API", version="2.0.0")

model = mlflow.pyfunc.load_model("models:/fraud-detection-best-model/Production")
print("Model loaded from MLflow Production registry!")

class Transaction(BaseModel):
    is_sell: int
    amount_scaled: float
    volume_scaled: float
    hour: int
    minute: int

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!", "data_source": "Kraken XBTUSD"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction):
    input_df = pd.DataFrame([{
        "amount_scaled": transaction.amount_scaled,
        "volume_scaled": transaction.volume_scaled,
        "hour":          transaction.hour,
        "minute":        transaction.minute,
        "is_sell":       transaction.is_sell,
    }])
    prediction = model.predict(input_df)[0]
    return {
        "prediction": int(prediction),
        "label": "FRAUD" if prediction == 1 else "LEGITIMATE"
    }