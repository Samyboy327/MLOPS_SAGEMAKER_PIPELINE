import boto3
import pandas as pd
from io import BytesIO

BUCKET_NAME = "rohit-telecom-churn-data-2026"

INPUT_KEY = "raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_KEY = "processed/customer_clean.csv"

s3 = boto3.client("s3")


# 1. Download raw data from S3 into memory
response = s3.get_object(
    Bucket=BUCKET_NAME,
    Key=INPUT_KEY
)

df = pd.read_csv(BytesIO(response["Body"].read()))

print("Raw data shape:", df.shape)


# 2. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# 3. Remove customerID
df = df.drop(columns=["customerID"])


# 4. Upload processed data directly to S3
csv_buffer = BytesIO()

df.to_csv(csv_buffer, index=False)

s3.put_object(
    Bucket=BUCKET_NAME,
    Key=OUTPUT_KEY,
    Body=csv_buffer.getvalue()
)

print("Processed data uploaded successfully!")
print("Output:", f"s3://{BUCKET_NAME}/{OUTPUT_KEY}")
print("Processed data shape:", df.shape)