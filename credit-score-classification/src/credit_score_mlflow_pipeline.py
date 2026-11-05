"""
SageMaker-friendly Credit Score Training Pipeline with MLflow
============================================================

Run:
    python credit_score_mlflow_pipeline.py --data_path data_B.csv

This version avoids saving custom preprocessing objects inside the model bundle.
Instead, it saves only standard Python objects:
    - trained RandomForestClassifier
    - fitted LabelEncoder
    - preprocessing_state dictionary

That makes the saved artifact easier to load inside SageMaker.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


@dataclass
class TrainingConfig:
    data_path: str
    experiment_name: str = "CreditScore_Final_RF"
    test_size: float = 0.2
    random_state: int = 42
    output_dir: str = "outputs"
    payment_behaviour_strategy: str = "onehot"


class CreditScorePreprocessor:
    """Fit/transform preprocessing, but save deployment state as plain dict."""

    drop_columns = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"]

    numeric_columns = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
        "Amount_invested_monthly", "Monthly_Balance",
    ]

    missing_indicator_columns = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_of_Loan",
        "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance",
        "Credit_History_Age_Months",
    ]

    median_columns = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
        "Amount_invested_monthly", "Monthly_Balance", "Credit_History_Age_Months",
    ]

    credit_mix_map = {"Bad": 0, "Standard": 1, "Good": 2}

    spending_map = {
        "Low_spent_Small_value_payments": "Low",
        "Low_spent_Medium_value_payments": "Low",
        "Low_spent_Large_value_payments": "Low",
        "High_spent_Small_value_payments": "High",
        "High_spent_Medium_value_payments": "High",
        "High_spent_Large_value_payments": "High",
    }

    payment_size_map = {
        "Low_spent_Small_value_payments": "Small",
        "Low_spent_Medium_value_payments": "Medium",
        "Low_spent_Large_value_payments": "Large",
        "High_spent_Small_value_payments": "Small",
        "High_spent_Medium_value_payments": "Medium",
        "High_spent_Large_value_payments": "Large",
    }

    def __init__(self, payment_behaviour_strategy: str = "onehot") -> None:
        if payment_behaviour_strategy not in {"onehot", "split"}:
            raise ValueError("payment_behaviour_strategy must be 'onehot' or 'split'.")
        self.payment_behaviour_strategy = payment_behaviour_strategy
        self.medians_: Dict[str, float] = {}
        self.loan_types_: List[str] = []
        self.feature_columns_: List[str] = []

    @staticmethod
    def _safe_median(series: pd.Series) -> float:
        non_null = series.dropna()
        return 0.0 if non_null.empty else float(non_null.median())

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = self._base_cleaning(X.copy(), fit=True)
        X_processed = self._encode_features(X_processed)
        X_processed = self._feature_engineering(X_processed)
        X_processed = self._final_cleaning(X_processed, fit=True)
        self.feature_columns_ = list(X_processed.columns)
        return X_processed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = self._base_cleaning(X.copy(), fit=False)
        X_processed = self._encode_features(X_processed)
        X_processed = self._feature_engineering(X_processed)
        X_processed = self._final_cleaning(X_processed, fit=False)
        X_processed = X_processed.reindex(columns=self.feature_columns_, fill_value=0)
        return X_processed

    def _base_cleaning(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = X.drop(columns=[c for c in self.drop_columns if c in X.columns])

        for col in self.numeric_columns:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col].astype(str).str.replace("_", "", regex=False), errors="coerce")

        if "Credit_History_Age" in X.columns:
            extracted = X["Credit_History_Age"].astype(str).str.extract(
                r"(?P<years>\d+)\s*Years?(?:\s*and\s*)?(?P<months>\d+)?\s*Months?"
            ).astype(float)
            X["Credit_History_Age_Months"] = extracted["years"].fillna(0) * 12 + extracted["months"].fillna(0)
            X = X.drop(columns=["Credit_History_Age"])

        for col in self.missing_indicator_columns:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isna().astype(int)

        X = self._handle_invalid_values(X, fit=fit)

        if fit:
            for col in self.median_columns:
                if col in X.columns:
                    self.medians_[col] = self._safe_median(X[col])

        for col, median_value in self.medians_.items():
            if col in X.columns and not col.startswith("final_") and not col.startswith("invalid_"):
                X[col] = X[col].fillna(median_value)

        X = self._process_type_of_loan(X, fit=fit)
        return X

    def _handle_invalid_values(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
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
                if fit:
                    self.medians_[f"invalid_{col}"] = self._safe_median(X[col])
                X[col] = X[col].fillna(self.medians_.get(f"invalid_{col}", 0.0))
        return X

    def _process_type_of_loan(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if "Type_of_Loan" not in X.columns:
            return X
        loan_series = (
            X["Type_of_Loan"].fillna("not specified").astype(str).str.lower().str.strip()
            .str.replace(" and ", ", ", regex=False)
        )
        if fit:
            loan_types = loan_series.str.split(",").explode().str.strip()
            loan_types = loan_types.replace({"nan": "not specified", "unknown": "not specified"})
            self.loan_types_ = sorted([loan for loan in loan_types.unique() if loan])
        for loan in self.loan_types_:
            safe_name = loan.replace(" ", "_").replace("/", "_").replace("-", "_")
            X[f"loan_{safe_name}"] = loan_series.str.contains(loan, case=False, na=False, regex=False).astype(int)
        return X.drop(columns=["Type_of_Loan"])

    def _encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if "Occupation" in X.columns:
            X["Occupation"] = X["Occupation"].replace("_______", "Unknown")
        if "Credit_Mix" in X.columns:
            X["Credit_Mix"] = X["Credit_Mix"].replace("_", np.nan).fillna("Standard")
            X["Credit_Mix"] = X["Credit_Mix"].map(self.credit_mix_map).fillna(1)
        if "Payment_of_Min_Amount" in X.columns:
            X["Payment_of_Min_Amount"] = X["Payment_of_Min_Amount"].replace("NM", "Unknown")
        if "Payment_Behaviour" in X.columns:
            X["Payment_Behaviour"] = X["Payment_Behaviour"].replace("!@9#%8", "Unknown")
        if self.payment_behaviour_strategy == "split" and "Payment_Behaviour" in X.columns:
            X["Spending_Level"] = X["Payment_Behaviour"].map(self.spending_map).fillna("Unknown")
            X["Payment_Size"] = X["Payment_Behaviour"].map(self.payment_size_map).fillna("Unknown")
            X["Spending_Level"] = X["Spending_Level"].map({"Low": 0, "High": 1, "Unknown": -1})
            X["Payment_Size"] = X["Payment_Size"].map({"Small": 0, "Medium": 1, "Large": 2, "Unknown": -1})
            X = X.drop(columns=["Payment_Behaviour"])
        categorical_cols = [c for c in ["Month", "Occupation", "Payment_of_Min_Amount", "Payment_Behaviour"] if c in X.columns]
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
        return X

    def _feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-6
        if {"Outstanding_Debt", "Annual_Income"}.issubset(X.columns):
            X["Debt_to_Income"] = X["Outstanding_Debt"] / (X["Annual_Income"] + eps)
        if {"Total_EMI_per_month", "Monthly_Inhand_Salary"}.issubset(X.columns):
            X["EMI_to_Salary"] = X["Total_EMI_per_month"] / (X["Monthly_Inhand_Salary"] + eps)
        if {"Num_Credit_Card", "Num_Bank_Accounts"}.issubset(X.columns):
            X["Card_Account_Ratio"] = X["Num_Credit_Card"] / (X["Num_Bank_Accounts"] + eps)
        if {"Num_of_Delayed_Payment", "Num_of_Loan"}.issubset(X.columns):
            X["Delay_per_Loan"] = X["Num_of_Delayed_Payment"] / (X["Num_of_Loan"] + 1)
        return X

    def _final_cleaning(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if fit:
            for col in numeric_cols:
                self.medians_[f"final_{col}"] = self._safe_median(X[col])
        for col in numeric_cols:
            X[col] = X[col].fillna(self.medians_.get(f"final_{col}", 0.0))
        return X

    def export_state(self) -> Dict[str, Any]:
        return {
            "payment_behaviour_strategy": self.payment_behaviour_strategy,
            "medians": self.medians_,
            "loan_types": self.loan_types_,
            "feature_columns": self.feature_columns_,
        }


class ModelTrainer:
    best_params = {
        "class_weight": None,
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 500,
    }

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> RandomForestClassifier:
        model = RandomForestClassifier(**self.best_params, random_state=self.random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        return model


class ModelEvaluator:
    @staticmethod
    def evaluate(model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        return {
            "metrics": {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            },
            "report_dict": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            "report_text": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }


class MLflowTrainingPipeline:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        self.preprocessor = CreditScorePreprocessor(config.payment_behaviour_strategy)
        self.trainer = ModelTrainer(config.random_state)
        self.evaluator = ModelEvaluator()
        self.label_encoder = LabelEncoder()

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.config.data_path}")
        df = pd.read_csv(self.config.data_path)
        if "Credit_Score" not in df.columns:
            raise ValueError("Dataset must contain target column: Credit_Score")
        return df

    def run(self) -> None:
        mlflow.set_experiment(self.config.experiment_name)
        df = self.load_data()
        X = df.drop(columns=["Credit_Score"])
        y = df["Credit_Score"]

        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=self.config.test_size, stratify=y, random_state=self.config.random_state
        )

        y_train = self.label_encoder.fit_transform(y_train_raw)
        y_test = self.label_encoder.transform(y_test_raw)
        X_train = self.preprocessor.fit_transform(X_train_raw)
        X_test = self.preprocessor.transform(X_test_raw)

        with mlflow.start_run(run_name="Final_RandomForest_SageMaker_Ready"):
            mlflow.log_param("model_name", "RandomForestClassifier")
            mlflow.log_param("payment_behaviour_strategy", self.config.payment_behaviour_strategy)
            mlflow.log_param("test_size", self.config.test_size)
            mlflow.log_param("random_state", self.config.random_state)
            mlflow.log_param("n_features", X_train.shape[1])
            for param_name, param_value in ModelTrainer.best_params.items():
                mlflow.log_param(param_name, param_value)

            model = self.trainer.train(X_train, y_train)
            result = self.evaluator.evaluate(model, X_test, y_test)
            mlflow.log_metrics(result["metrics"])

            report_txt = os.path.join(self.config.output_dir, "classification_report.txt")
            report_json = os.path.join(self.config.output_dir, "classification_report.json")
            cm_csv = os.path.join(self.config.output_dir, "confusion_matrix.csv")
            cm_png = os.path.join(self.config.output_dir, "confusion_matrix.png")
            model_path = os.path.join(self.config.output_dir, "best_credit_score_model.pkl")

            with open(report_txt, "w", encoding="utf-8") as f:
                f.write(result["report_text"])
            with open(report_json, "w", encoding="utf-8") as f:
                json.dump(result["report_dict"], f, indent=4)
            pd.DataFrame(result["confusion_matrix"]).to_csv(cm_csv, index=False)

            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(result["confusion_matrix"]).plot(ax=ax, values_format="d")
            ax.set_title("Confusion Matrix - Final Random Forest")
            plt.tight_layout()
            fig.savefig(cm_png)
            plt.close(fig)

            model_bundle = {
                "model": model,
                "label_encoder": self.label_encoder,
                "preprocessing_state": self.preprocessor.export_state(),
                "model_name": "RandomForestClassifier",
                "model_params": ModelTrainer.best_params,
                "metrics": result["metrics"],
            }
            joblib.dump(model_bundle, model_path)

            for artifact in [report_txt, report_json, cm_csv, cm_png, model_path]:
                mlflow.log_artifact(artifact)
            mlflow.sklearn.log_model(model, artifact_path="random_forest_model")

        print("\nTraining selesai.")
        print("Saved model bundle:", model_path)
        print("Metrics:")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification report:")
        print(result["report_text"])


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="SageMaker-friendly Credit Score MLflow pipeline")
    parser.add_argument("--data_path", type=str, default="data_B.csv")
    parser.add_argument("--experiment_name", type=str, default="CreditScore_Final_RF")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--payment_behaviour_strategy", choices=["onehot", "split"], default="onehot")
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


if __name__ == "__main__":
    MLflowTrainingPipeline(parse_args()).run()
