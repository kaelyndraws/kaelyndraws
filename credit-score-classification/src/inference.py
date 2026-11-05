"""
SageMaker-compatible inference script for Credit Score Classification.

SageMaker functions included:
    - model_fn(model_dir)
    - input_fn(request_body, content_type)
    - predict_fn(input_data, model_bundle)
    - output_fn(prediction, accept)

The model bundle must be created by credit_score_mlflow_pipeline.py and saved as:
    best_credit_score_model.pkl
"""

from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd


CREDIT_MIX_MAP = {"Bad": 0, "Standard": 1, "Good": 2}

SPENDING_MAP = {
    "Low_spent_Small_value_payments": "Low",
    "Low_spent_Medium_value_payments": "Low",
    "Low_spent_Large_value_payments": "Low",
    "High_spent_Small_value_payments": "High",
    "High_spent_Medium_value_payments": "High",
    "High_spent_Large_value_payments": "High",
}

PAYMENT_SIZE_MAP = {
    "Low_spent_Small_value_payments": "Small",
    "Low_spent_Medium_value_payments": "Medium",
    "Low_spent_Large_value_payments": "Large",
    "High_spent_Small_value_payments": "Small",
    "High_spent_Medium_value_payments": "Medium",
    "High_spent_Large_value_payments": "Large",
}

DROP_COLUMNS = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"]

NUMERIC_COLUMNS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance",
]

MISSING_INDICATOR_COLUMNS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_of_Loan",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance",
    "Credit_History_Age_Months",
]


def _safe_int(value: Any) -> Union[int, str]:
    try:
        return int(value)
    except Exception:
        return str(value)


def preprocess_input(raw_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame], state: Dict[str, Any]) -> pd.DataFrame:
    """Apply the same preprocessing used during training using saved plain state."""
    if isinstance(raw_data, pd.DataFrame):
        X = raw_data.copy()
    elif isinstance(raw_data, list):
        X = pd.DataFrame(raw_data)
    elif isinstance(raw_data, dict):
        X = pd.DataFrame([raw_data])
    else:
        raise ValueError("Input must be a dict, list of dicts, or pandas DataFrame.")

    medians: Dict[str, float] = state.get("medians", {})
    loan_types: List[str] = state.get("loan_types", [])
    feature_columns: List[str] = state.get("feature_columns", [])
    payment_behaviour_strategy = state.get("payment_behaviour_strategy", "onehot")

    X = X.drop(columns=[c for c in DROP_COLUMNS if c in X.columns])

    for col in NUMERIC_COLUMNS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col].astype(str).str.replace("_", "", regex=False), errors="coerce")

    if "Credit_History_Age" in X.columns:
        extracted = X["Credit_History_Age"].astype(str).str.extract(
            r"(?P<years>\d+)\s*Years?(?:\s*and\s*)?(?P<months>\d+)?\s*Months?"
        ).astype(float)
        X["Credit_History_Age_Months"] = extracted["years"].fillna(0) * 12 + extracted["months"].fillna(0)
        X = X.drop(columns=["Credit_History_Age"])

    for col in MISSING_INDICATOR_COLUMNS:
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype(int)

    invalid_rules = {
        "Age": lambda s: (s < 18) | (s > 120),
        "Num_Bank_Accounts": lambda s: (s < 0) | (s > 50),
        "Num_Credit_Card": lambda s: (s < 0) | (s > 100),
        "Interest_Rate": lambda s: (s < 0) | (s > 200),
        "Num_of_Loan": lambda s: (s < 0) | (s > 100),
        "Delay_from_due_date": lambda s: s < 0,
        "Num_Credit_Inquiries": lambda s: (s < 0) | (s > 100),
        "Num_of_Delayed_Payment": lambda s: s < 0,
        "Total_EMI_per_month": lambda s: s < 0,
    }
    for col, rule in invalid_rules.items():
        if col in X.columns:
            X.loc[rule(X[col]), col] = np.nan
            X[col] = X[col].fillna(medians.get(f"invalid_{col}", medians.get(col, 0.0)))

    for col, median_value in medians.items():
        if col in X.columns and not col.startswith("final_") and not col.startswith("invalid_"):
            X[col] = X[col].fillna(median_value)

    if "Type_of_Loan" in X.columns:
        loan_series = (
            X["Type_of_Loan"].fillna("not specified").astype(str).str.lower().str.strip()
            .str.replace(" and ", ", ", regex=False)
        )
        for loan in loan_types:
            safe_name = loan.replace(" ", "_").replace("/", "_").replace("-", "_")
            X[f"loan_{safe_name}"] = loan_series.str.contains(loan, case=False, na=False, regex=False).astype(int)
        X = X.drop(columns=["Type_of_Loan"])

    if "Occupation" in X.columns:
        X["Occupation"] = X["Occupation"].replace("_______", "Unknown")
    if "Credit_Mix" in X.columns:
        X["Credit_Mix"] = X["Credit_Mix"].replace("_", np.nan).fillna("Standard")
        X["Credit_Mix"] = X["Credit_Mix"].map(CREDIT_MIX_MAP).fillna(1)
    if "Payment_of_Min_Amount" in X.columns:
        X["Payment_of_Min_Amount"] = X["Payment_of_Min_Amount"].replace("NM", "Unknown")
    if "Payment_Behaviour" in X.columns:
        X["Payment_Behaviour"] = X["Payment_Behaviour"].replace("!@9#%8", "Unknown")

    if payment_behaviour_strategy == "split" and "Payment_Behaviour" in X.columns:
        X["Spending_Level"] = X["Payment_Behaviour"].map(SPENDING_MAP).fillna("Unknown")
        X["Payment_Size"] = X["Payment_Behaviour"].map(PAYMENT_SIZE_MAP).fillna("Unknown")
        X["Spending_Level"] = X["Spending_Level"].map({"Low": 0, "High": 1, "Unknown": -1})
        X["Payment_Size"] = X["Payment_Size"].map({"Small": 0, "Medium": 1, "Large": 2, "Unknown": -1})
        X = X.drop(columns=["Payment_Behaviour"])

    categorical_cols = [c for c in ["Month", "Occupation", "Payment_of_Min_Amount", "Payment_Behaviour"] if c in X.columns]
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

    eps = 1e-6
    if {"Outstanding_Debt", "Annual_Income"}.issubset(X.columns):
        X["Debt_to_Income"] = X["Outstanding_Debt"] / (X["Annual_Income"] + eps)
    if {"Total_EMI_per_month", "Monthly_Inhand_Salary"}.issubset(X.columns):
        X["EMI_to_Salary"] = X["Total_EMI_per_month"] / (X["Monthly_Inhand_Salary"] + eps)
    if {"Num_Credit_Card", "Num_Bank_Accounts"}.issubset(X.columns):
        X["Card_Account_Ratio"] = X["Num_Credit_Card"] / (X["Num_Bank_Accounts"] + eps)
    if {"Num_of_Delayed_Payment", "Num_of_Loan"}.issubset(X.columns):
        X["Delay_per_Loan"] = X["Num_of_Delayed_Payment"] / (X["Num_of_Loan"] + 1)

    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(medians.get(f"final_{col}", 0.0))

    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


class CreditScorePredictor:
    """Local/FastAPI helper that uses the same logic as SageMaker."""

    def __init__(self, model_path: str = "outputs/best_credit_score_model.pkl") -> None:
        self.bundle = joblib.load(model_path)

    def predict_one(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return predict_fn(data, self.bundle)


# ---------- SageMaker serving functions ----------

def model_fn(model_dir: str) -> Dict[str, Any]:
    model_path = os.path.join(model_dir, "best_credit_score_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bundle not found at: {model_path}")
    return joblib.load(model_path)


def input_fn(request_body: str, content_type: str) -> Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    if content_type == "application/json":
        return json.loads(request_body)
    if content_type == "text/csv":
        return pd.read_csv(io.StringIO(request_body))
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame], model_bundle: Dict[str, Any]) -> Dict[str, Any]:
    model = model_bundle["model"]
    label_encoder = model_bundle.get("label_encoder")
    state = model_bundle["preprocessing_state"]

    X_processed = preprocess_input(input_data, state)
    encoded_predictions = model.predict(X_processed)

    if label_encoder is not None:
        labels = label_encoder.inverse_transform(encoded_predictions.astype(int))
    else:
        labels = encoded_predictions.astype(str)

    result: Dict[str, Any] = {
        "predictions": [str(x) for x in labels],
        "prediction_encoded": [_safe_int(x) for x in encoded_predictions],
        "model_name": model_bundle.get("model_name", "Unknown"),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_processed)
        class_names = [str(c) for c in label_encoder.classes_] if label_encoder is not None else [str(i) for i in range(probabilities.shape[1])]
        result["probabilities"] = [
            {class_name: float(prob) for class_name, prob in zip(class_names, row)}
            for row in probabilities
        ]

    if len(result["predictions"]) == 1:
        result["prediction"] = result["predictions"][0]
        result["probability"] = result.get("probabilities", [None])[0]

    return result


def output_fn(prediction: Dict[str, Any], accept: str) -> str:
    if accept in ("application/json", "*/*"):
        return json.dumps(prediction)
    raise ValueError(f"Unsupported accept type: {accept}")
