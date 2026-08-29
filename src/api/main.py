"""
src/api/main.py
---------------
FastAPI REST endpoint for credit default prediction.
Accepts customer financial data and returns default probability + risk factors.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Load model and features ────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best_model.joblib"
FEATURES_PATH = Path(__file__).resolve().parents[2] / "models" / "feature_names.joblib"

try:
    model = joblib.load(MODEL_PATH)
    FEATURES = joblib.load(FEATURES_PATH)
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"✅ Features: {FEATURES}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    FEATURES = []

# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="Ethiopian Fintech Credit Risk API",
    description="Predicts credit default probability using XGBoost model trained on Give Me Some Credit dataset.",
    version="1.0.0"
)


# ── Request schema ─────────────────────────────────────────────
class CustomerData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0, le=1, description="Credit utilization ratio (0-1)"
    )
    age: int = Field(..., ge=18, le=96, description="Customer age")
    DebtRatio: float = Field(..., ge=0, le=1, description="Debt ratio (0-1)")
    MonthlyIncome: float = Field(..., ge=0, description="Monthly income in USD")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, description="Number of open credit lines")
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, description="Number of real estate loans")
    delinquency_score: float = Field(..., ge=0, description="Weighted delinquency score")
    debt_to_income: float = Field(..., ge=0, description="Debt to income ratio")
    is_young_borrower: int = Field(..., ge=0, le=1, description="1 if age < 30")
    is_senior_borrower: int = Field(..., ge=0, le=1, description="1 if age >= 60")
    high_utilization: int = Field(..., ge=0, le=1, description="1 if utilization > 0.8")
    total_past_due: float = Field(..., ge=0, description="Total past due events")
    has_delinquency: int = Field(..., ge=0, le=1, description="1 if any delinquency")
    has_dependents: int = Field(..., ge=0, le=1, description="1 if has dependents")

    class Config:
        json_schema_extra = {
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.75,
                "age": 35,
                "DebtRatio": 0.45,
                "MonthlyIncome": 5000,
                "NumberOfOpenCreditLinesAndLoans": 4,
                "NumberRealEstateLoansOrLines": 1,
                "delinquency_score": 0,
                "debt_to_income": 2250,
                "is_young_borrower": 0,
                "is_senior_borrower": 0,
                "high_utilization": 0,
                "total_past_due": 0,
                "has_delinquency": 0,
                "has_dependents": 1
            }
        }


# ── Response schema ────────────────────────────────────────────
class PredictionResponse(BaseModel):
    default_probability: float
    risk_tier: str
    prediction: str
    top_risk_factors: list
    model: str


# ── Helper functions ───────────────────────────────────────────
def get_risk_tier(probability: float) -> str:
    if probability < 0.10:
        return "Low Risk"
    elif probability < 0.25:
        return "Moderate Risk"
    elif probability < 0.50:
        return "High Risk"
    else:
        return "Very High Risk"


def get_top_risk_factors(input_data: dict) -> list:
    """Return top risk factors based on input values."""
    factors = []
    if input_data.get("has_delinquency") == 1:
        factors.append("Past delinquency history detected")
    if input_data.get("high_utilization") == 1:
        factors.append("High credit utilization (>80%)")
    if input_data.get("delinquency_score", 0) > 3:
        factors.append("Severe delinquency score")
    if input_data.get("is_young_borrower") == 1:
        factors.append("Young borrower (<30) — higher risk group")
    if input_data.get("DebtRatio", 0) > 0.7:
        factors.append("High debt ratio (>70%)")
    if input_data.get("MonthlyIncome", 0) < 2000:
        factors.append("Low monthly income")
    if not factors:
        factors.append("No major risk factors identified")
    return factors[:3]


# ── Routes ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Ethiopian Fintech Credit Risk API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Build input DataFrame in correct feature order
    input_dict = customer.dict()
    input_df = pd.DataFrame([input_dict])[FEATURES]

    # Predict
    probability = float(model.predict_proba(input_df)[0][1])
    prediction = "DEFAULT" if probability >= 0.5 else "NO DEFAULT"
    risk_tier = get_risk_tier(probability)
    top_factors = get_top_risk_factors(input_dict)

    return PredictionResponse(
        default_probability=round(probability, 4),
        risk_tier=risk_tier,
        prediction=prediction,
        top_risk_factors=top_factors,
        model=type(model).__name__
    )


# ── Run locally ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
