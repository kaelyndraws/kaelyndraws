"""
Streamlit Frontend for Credit Score Classification via SageMaker Endpoint
========================================================================

This version calls an AWS SageMaker real-time endpoint directly using boto3.
It does not call FastAPI and does not load the model locally.

Run:
    streamlit run streamlit_credit_score_sagemaker_app.py

Requirements:
    pip install streamlit pandas boto3

Make sure your AWS credentials/role can call:
    sagemaker-runtime:InvokeEndpoint
"""

from __future__ import annotations

import json
from typing import Any, Dict

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError


DEFAULT_ENDPOINT_NAME = "credit-score-endpoint"
DEFAULT_REGION = "us-east-1"

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August"
]

OCCUPATIONS = [
    "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", "Lawyer",
    "Media_Manager", "Doctor", "Journalist", "Manager", "Accountant", "Musician",
    "Mechanic", "Writer", "Architect", "Unknown"
]

CREDIT_MIX_OPTIONS = ["Bad", "Standard", "Good", "_"]
PAYMENT_MIN_OPTIONS = ["No", "Yes", "NM"]
PAYMENT_BEHAVIOUR_OPTIONS = [
    "Low_spent_Small_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments",
    "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments",
    "!@9#%8",
]

LOAN_TYPES = [
    "Auto Loan",
    "Credit-Builder Loan",
    "Debt Consolidation Loan",
    "Home Equity Loan",
    "Mortgage Loan",
    "Not Specified",
    "Payday Loan",
    "Personal Loan",
    "Student Loan",
]

TEST_CASES: Dict[str, Dict[str, Any]] = {
    "Poor-like customer": {
        "Month": "July",
        "Age": 25,
        "Occupation": "Developer",
        "Annual_Income": 24000.0,
        "Monthly_Inhand_Salary": 1800.0,
        "Num_Bank_Accounts": 9,
        "Num_Credit_Card": 8,
        "Interest_Rate": 29,
        "Num_of_Loan": 8,
        "Type_of_Loan": "Payday Loan, Personal Loan, Debt Consolidation Loan",
        "Delay_from_due_date": 45,
        "Num_of_Delayed_Payment": 18,
        "Changed_Credit_Limit": 23.0,
        "Num_Credit_Inquiries": 12,
        "Credit_Mix": "Bad",
        "Outstanding_Debt": 4200.0,
        "Credit_Utilization_Ratio": 42.0,
        "Credit_History_Age": "2 Years and 3 Months",
        "Payment_of_Min_Amount": "Yes",
        "Total_EMI_per_month": 560.0,
        "Amount_invested_monthly": 35.0,
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Monthly_Balance": 120.0,
    },
    "Standard-like customer": {
        "Month": "May",
        "Age": 35,
        "Occupation": "Teacher",
        "Annual_Income": 58000.0,
        "Monthly_Inhand_Salary": 4200.0,
        "Num_Bank_Accounts": 5,
        "Num_Credit_Card": 5,
        "Interest_Rate": 14,
        "Num_of_Loan": 3,
        "Type_of_Loan": "Auto Loan, Credit-Builder Loan",
        "Delay_from_due_date": 12,
        "Num_of_Delayed_Payment": 6,
        "Changed_Credit_Limit": 8.0,
        "Num_Credit_Inquiries": 5,
        "Credit_Mix": "Standard",
        "Outstanding_Debt": 1200.0,
        "Credit_Utilization_Ratio": 32.0,
        "Credit_History_Age": "14 Years and 6 Months",
        "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 180.0,
        "Amount_invested_monthly": 180.0,
        "Payment_Behaviour": "High_spent_Medium_value_payments",
        "Monthly_Balance": 420.0,
    },
    "Good-like customer": {
        "Month": "March",
        "Age": 44,
        "Occupation": "Scientist",
        "Annual_Income": 98000.0,
        "Monthly_Inhand_Salary": 7800.0,
        "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 3,
        "Interest_Rate": 6,
        "Num_of_Loan": 1,
        "Type_of_Loan": "Mortgage Loan",
        "Delay_from_due_date": 2,
        "Num_of_Delayed_Payment": 1,
        "Changed_Credit_Limit": 2.0,
        "Num_Credit_Inquiries": 1,
        "Credit_Mix": "Good",
        "Outstanding_Debt": 350.0,
        "Credit_Utilization_Ratio": 25.0,
        "Credit_History_Age": "25 Years and 8 Months",
        "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 90.0,
        "Amount_invested_monthly": 650.0,
        "Payment_Behaviour": "High_spent_Large_value_payments",
        "Monthly_Balance": 950.0,
    },
}


def _default(case: Dict[str, Any], key: str, fallback: Any) -> Any:
    return case.get(key, fallback)


def _history_defaults(defaults: Dict[str, Any]) -> tuple[int, int]:
    raw = str(defaults.get("Credit_History_Age", "14 Years and 6 Months"))
    years, months = 14, 6
    try:
        if "Years" in raw:
            years = int(raw.split("Years")[0].strip())
        if "and" in raw and "Months" in raw:
            months = int(raw.split("and")[1].split("Months")[0].strip())
    except Exception:
        pass
    return years, months


def build_form(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Collect raw customer input using the same column names as the training dataset."""
    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        month = st.selectbox("Month", MONTHS, index=MONTHS.index(_default(defaults, "Month", "July")))
        age = st.number_input("Age", min_value=18, max_value=120, value=int(_default(defaults, "Age", 35)))
        occupation_default = _default(defaults, "Occupation", "Developer")
        occupation = st.selectbox(
            "Occupation",
            OCCUPATIONS,
            index=OCCUPATIONS.index(occupation_default) if occupation_default in OCCUPATIONS else OCCUPATIONS.index("Unknown"),
        )
        annual_income = st.number_input("Annual Income", min_value=0.0, value=float(_default(defaults, "Annual_Income", 50000.0)))
        monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=float(_default(defaults, "Monthly_Inhand_Salary", 4000.0)))
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=50, value=int(_default(defaults, "Num_Bank_Accounts", 5)))
        num_credit_card = st.number_input("Number of Credit Cards", min_value=0, max_value=50, value=int(_default(defaults, "Num_Credit_Card", 5)))

    with col2:
        interest_rate = st.number_input("Interest Rate", min_value=0, max_value=200, value=int(_default(defaults, "Interest_Rate", 14)))
        num_of_loan = st.number_input("Number of Loans", min_value=0, max_value=100, value=int(_default(defaults, "Num_of_Loan", 3)))
        selected_loans = st.multiselect(
            "Type of Loan",
            LOAN_TYPES,
            default=[loan.strip() for loan in str(_default(defaults, "Type_of_Loan", "Not Specified")).split(",") if loan.strip()],
        )
        delay_from_due_date = st.number_input("Delay from Due Date", min_value=0, max_value=365, value=int(_default(defaults, "Delay_from_due_date", 12)))
        delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, max_value=200, value=int(_default(defaults, "Num_of_Delayed_Payment", 6)))
        changed_limit = st.number_input("Changed Credit Limit", min_value=-50.0, max_value=100.0, value=float(_default(defaults, "Changed_Credit_Limit", 8.0)))
        credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, max_value=100, value=int(_default(defaults, "Num_Credit_Inquiries", 5)))

    with col3:
        credit_mix = st.selectbox("Credit Mix", CREDIT_MIX_OPTIONS, index=CREDIT_MIX_OPTIONS.index(_default(defaults, "Credit_Mix", "Standard")))
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=float(_default(defaults, "Outstanding_Debt", 1200.0)))
        credit_utilization = st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=100.0, value=float(_default(defaults, "Credit_Utilization_Ratio", 32.0)))
        default_years, default_months = _history_defaults(defaults)
        history_years = st.number_input("Credit History - Years", min_value=0, max_value=60, value=default_years)
        history_months = st.number_input("Credit History - Months", min_value=0, max_value=11, value=default_months)
        payment_min = st.selectbox("Payment of Minimum Amount", PAYMENT_MIN_OPTIONS, index=PAYMENT_MIN_OPTIONS.index(_default(defaults, "Payment_of_Min_Amount", "No")))
        total_emi = st.number_input("Total EMI per Month", min_value=0.0, value=float(_default(defaults, "Total_EMI_per_month", 180.0)))
        invested = st.number_input("Amount Invested Monthly", min_value=0.0, value=float(_default(defaults, "Amount_invested_monthly", 180.0)))
        payment_behaviour = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR_OPTIONS, index=PAYMENT_BEHAVIOUR_OPTIONS.index(_default(defaults, "Payment_Behaviour", "High_spent_Medium_value_payments")))
        monthly_balance = st.number_input("Monthly Balance", min_value=0.0, value=float(_default(defaults, "Monthly_Balance", 420.0)))

    return {
        "Month": month,
        "Age": age,
        "Occupation": occupation if occupation != "Unknown" else "_______",
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan,
        "Type_of_Loan": ", ".join(selected_loans) if selected_loans else "Not Specified",
        "Delay_from_due_date": delay_from_due_date,
        "Num_of_Delayed_Payment": delayed_payment,
        "Changed_Credit_Limit": changed_limit,
        "Num_Credit_Inquiries": credit_inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization,
        "Credit_History_Age": f"{history_years} Years and {history_months} Months",
        "Payment_of_Min_Amount": payment_min,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": invested,
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": monthly_balance,
    }


def get_sagemaker_runtime(region_name: str):
    return boto3.client("sagemaker-runtime", region_name=region_name)


def call_sagemaker_endpoint(endpoint_name: str, region_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke SageMaker real-time endpoint with one raw customer payload."""
    runtime = get_sagemaker_runtime(region_name)

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )

    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


def extract_prediction(result: Dict[str, Any]) -> str:
    """Support several common SageMaker inference response formats."""
    if "prediction" in result:
        return str(result["prediction"])
    if "label" in result:
        return str(result["label"])
    if "credit_score" in result:
        return str(result["credit_score"])
    if "class" in result:
        return str(result["class"])
    if "predictions" in result:
        predictions = result["predictions"]
        if isinstance(predictions, list) and predictions:
            first = predictions[0]
            if isinstance(first, dict):
                return str(first.get("prediction", first.get("label", first)))
            return str(first)
    return str(result)


def main() -> None:
    st.set_page_config(page_title="Credit Score Prediction", page_icon="💳", layout="wide")

    st.title("Credit Score Classification - SageMaker Deployment")
    st.write(
        "Aplikasi ini adalah frontend Streamlit. Input dikirim langsung ke SageMaker real-time endpoint, "
        "lalu endpoint menjalankan preprocessing dan model terbaik."
    )

    with st.sidebar:
        st.header("SageMaker Settings")
        endpoint_name = st.text_input("SageMaker Endpoint Name", value=DEFAULT_ENDPOINT_NAME)
        region_name = st.text_input("AWS Region", value=DEFAULT_REGION)
        st.caption("Contoh endpoint: credit-score-endpoint")

        st.header("Test Case")
        test_case_name = st.selectbox("Pilih test case", list(TEST_CASES.keys()))
        use_test_case = st.checkbox("Gunakan nilai test case", value=True)

    defaults = TEST_CASES[test_case_name] if use_test_case else {}
    features = build_form(defaults)

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Payload yang Dikirim ke SageMaker")
        st.json(features)

    with right:
        st.subheader("Prediction Result")
        if st.button("Predict Credit Score", type="primary"):
            try:
                result = call_sagemaker_endpoint(endpoint_name, region_name, features)
                prediction = extract_prediction(result)
                st.success(f"Predicted Credit Score: {prediction}")
                st.write("Raw SageMaker response:")
                st.json(result)
            except (BotoCoreError, ClientError) as err:
                st.error("Gagal memanggil SageMaker endpoint.")
                st.code(str(err))
            except json.JSONDecodeError as err:
                st.error("Endpoint mengembalikan response yang bukan JSON.")
                st.code(str(err))
            except Exception as err:
                st.error(f"Terjadi error: {err}")

    st.divider()
    st.info(
        "Untuk bukti deployment, jalankan minimal tiga test case: Poor-like, Standard-like, dan Good-like. "
        "Ambil screenshot pada hasil prediksi masing-masing kelas dan screenshot status endpoint di SageMaker."
    )


if __name__ == "__main__":
    main()
