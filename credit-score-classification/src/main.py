"""
Optional local FastAPI backend for Credit Score inference.

This is useful for local testing with Streamlit/FastAPI. SageMaker deployment uses
src/inference.py directly through model_fn/input_fn/predict_fn/output_fn.

Run:
    uvicorn main:app --reload
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import CreditScorePredictor


MODEL_PATH = "outputs/best_credit_score_model.pkl"

app = FastAPI(title="Credit Score Classification API")
predictor = CreditScorePredictor(model_path=MODEL_PATH)


class PredictionRequest(BaseModel):
    Month: str
    Age: float
    Occupation: str
    Annual_Income: float
    Monthly_Inhand_Salary: float
    Num_Bank_Accounts: int
    Num_Credit_Card: int
    Interest_Rate: float
    Num_of_Loan: int
    Type_of_Loan: str
    Delay_from_due_date: int
    Num_of_Delayed_Payment: int
    Changed_Credit_Limit: float
    Num_Credit_Inquiries: int
    Credit_Mix: str
    Outstanding_Debt: float
    Credit_Utilization_Ratio: float
    Credit_History_Age: str
    Payment_of_Min_Amount: str
    Total_EMI_per_month: float
    Amount_invested_monthly: float
    Payment_Behaviour: str
    Monthly_Balance: float


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Credit Score Classification API is running."}


@app.post("/predict")
def predict(data: PredictionRequest) -> Dict[str, Any]:
    try:
        return predictor.predict_one(data.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
