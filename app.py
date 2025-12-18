"""
Minimal FastAPI app for model inference.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path

# Load model at startup
MODEL_PATH = Path("models/model.joblib")

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None

app = FastAPI(title="ML Inference API")


class PredictionInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    income: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)

    class Config:
        json_schema_extra = {
            "example": {"age": 35, "income": 75000, "credit_score": 720}
        }


class PredictionOutput(BaseModel):
    prediction: int
    probability: float


@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to DataFrame (model expects this format)
    df = pd.DataFrame([data.model_dump()])
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0].max()
    
    return PredictionOutput(
        prediction=int(prediction),
        probability=float(probability)
    )
