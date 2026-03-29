import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API", version="1.0.0")

model = joblib.load("models/trained/fraud_model.pkl")

class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction):
    features = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return {
        "prediction": int(prediction),
        "label": "FRAUD" if prediction == 1 else "LEGITIMATE",
        "fraud_probability": round(float(probability), 4)
    }