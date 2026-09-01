import boto3
from pathlib import Path

BUCKET_NAME = "rohit-telecom-churn-data-2026"

LOCAL_FILE = Path(
    r"D:\MLOPS_S3_DATA_PIPELINE\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

S3_KEY = "raw/WA_Fn-UseC_-Telco-Customer-Churn-september.csv"

s3 = boto3.client("s3")

try:
    print(f"Uploading: {LOCAL_FILE}")
    print(f"Destination: s3://{BUCKET_NAME}/{S3_KEY}")

    s3.upload_file(
        str(LOCAL_FILE),
        BUCKET_NAME,
        S3_KEY
    )

    print("Dataset uploaded successfully!")
    print(f"s3://{BUCKET_NAME}/{S3_KEY}")

except Exception as e:
    print(f"Upload failed: {e}")
    raise