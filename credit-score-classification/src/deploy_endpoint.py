"""
deploy_endpoint.py
==================

Deploy the Credit Score model bundle to a SageMaker real-time endpoint.

Run after training:
    python credit_score_mlflow_pipeline.py --data_path data_B.csv
    python deploy_endpoint.py --delete_existing

This script will:
    1. Package outputs/best_credit_score_model.pkl into model.tar.gz
    2. Upload model.tar.gz to S3
    3. Create and deploy a SageMaker SKLearn endpoint
    4. Optionally invoke the endpoint with sample test cases
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel


BUCKET = "final-project-kaelyn-renata-halim-1501"
MODEL_S3_KEY = "credit-score/model.tar.gz"
ENDPOINT_NAME = "credit-score-endpoint"

LOCAL_MODEL_PATH = Path("outputs/best_credit_score_model.pkl")
LOCAL_INFERENCE_PATH = Path("src/inference.py")
LOCAL_REQUIREMENTS_PATH = Path("src/requirements.txt")
LOCAL_TAR_PATH = Path("model.tar.gz")

MODEL_URI = f"s3://{BUCKET}/{MODEL_S3_KEY}"


TEST_CASES: List[Dict[str, Any]] = [
    {
        "case_name": "Poor-like customer",
        "payload": {
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
    },
    {
        "case_name": "Standard-like customer",
        "payload": {
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
    },
    {
        "case_name": "Good-like customer",
        "payload": {
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
    },
]


def get_role() -> str:
    """Get SageMaker execution role from notebook environment or IAM fallback."""
    try:
        return sagemaker.get_execution_role()
    except Exception:
        iam = boto3.client("iam")
        return iam.get_role(RoleName="LabRole")["Role"]["Arn"]


def delete_existing_endpoint(endpoint_name: str) -> None:
    sm_client = boto3.client("sagemaker")
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Deleting existing endpoint: {endpoint_name}")
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        waiter = sm_client.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=endpoint_name)
    except sm_client.exceptions.ClientError:
        print(f"No existing endpoint found: {endpoint_name}")


def package_model() -> Path:
    """Create model.tar.gz containing the saved model bundle."""
    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {LOCAL_MODEL_PATH}. Run credit_score_mlflow_pipeline.py first."
        )
    if not LOCAL_INFERENCE_PATH.exists():
        raise FileNotFoundError(f"Missing {LOCAL_INFERENCE_PATH}.")

    if LOCAL_TAR_PATH.exists():
        LOCAL_TAR_PATH.unlink()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        shutil.copy2(LOCAL_MODEL_PATH, tmp_path / "best_credit_score_model.pkl")

        # Keeping code inside the tar is useful for backup/reproducibility.
        code_dir = tmp_path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LOCAL_INFERENCE_PATH, code_dir / "inference.py")
        if LOCAL_REQUIREMENTS_PATH.exists():
            shutil.copy2(LOCAL_REQUIREMENTS_PATH, code_dir / "requirements.txt")

        with tarfile.open(LOCAL_TAR_PATH, "w:gz") as tar:
            for item in tmp_path.iterdir():
                tar.add(item, arcname=item.name)

    print(f"Created: {LOCAL_TAR_PATH}")
    return LOCAL_TAR_PATH


def upload_model_to_s3(tar_path: Path) -> str:
    s3_client = boto3.client("s3")
    print(f"Uploading {tar_path} to {MODEL_URI}")
    s3_client.upload_file(str(tar_path), BUCKET, MODEL_S3_KEY)
    print("Upload complete.")
    return MODEL_URI


def deploy_sagemaker_endpoint(delete_existing: bool = False, instance_type: str = "ml.m5.large"):
    if delete_existing:
        delete_existing_endpoint(ENDPOINT_NAME)

    role = get_role()
    print(f"Using role: {role}")
    print(f"Using model URI: {MODEL_URI}")

    sklearn_model = SKLearnModel(
        model_data=MODEL_URI,
        role=role,
        entry_point="inference.py",
        source_dir="src",
        framework_version="1.2-1",
        py_version="py3",
    )

    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=ENDPOINT_NAME,
    )

    print(f"Endpoint deployed: {ENDPOINT_NAME}")
    return predictor


def invoke_test_cases() -> None:
    runtime = boto3.client("sagemaker-runtime")
    for test_case in TEST_CASES:
        print(f"\nTesting: {test_case['case_name']}")
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(test_case["payload"]),
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Credit Score model to SageMaker endpoint.")
    parser.add_argument("--delete_existing", action="store_true", help="Delete endpoint with same name before deploying.")
    parser.add_argument("--instance_type", default="ml.m5.large", help="SageMaker endpoint instance type.")
    parser.add_argument("--skip_package", action="store_true", help="Skip creating and uploading model.tar.gz.")
    parser.add_argument("--test", action="store_true", help="Invoke endpoint with sample test cases after deployment.")
    args = parser.parse_args()

    if not args.skip_package:
        tar_path = package_model()
        upload_model_to_s3(tar_path)

    deploy_sagemaker_endpoint(delete_existing=args.delete_existing, instance_type=args.instance_type)

    if args.test:
        invoke_test_cases()


if __name__ == "__main__":
    main()
