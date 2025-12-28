from fastapi import FastAPI
import joblib
import pandas as pd
from .schemas import CreditInput
from .model import predict

app = FastAPI(
    title="Credit Default Prediction API",
    description="Predict probability of credit default",
    version="1.0"
)

model = joblib.load('artifacts/XGB.pkl')

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict_default(input_data: CreditInput):
    input_dict = input_data.model_dump()
    probability = predict(input_dict)

    return {
        "default_probability": round(probability, 4),
        "risk_level": "High" if probability > 0.5 else "Low"
    }
