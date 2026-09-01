import pandas as pd

file_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nFirst 5 Rows:")
print(df.head())

print("\nBlank TotalCharges:")
print((df["TotalCharges"].str.strip() == "").sum())

print("\nRows with blank TotalCharges:")
print(df[df["TotalCharges"].str.strip() == ""])

df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nTotalCharges Data Type After Cleaning:")
print(df["TotalCharges"].dtype)

print("\nMissing TotalCharges After Cleaning:")
print(df["TotalCharges"].isnull().sum())

df = df.drop(columns = ["customerID"])

print(df.shape)

print("\nunique Churn values")
print(df["Churn"].unique())

print("\nUnique Values in Categorical Columns:")

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].unique())

output_path = "data/processed/customer_clean.csv"

df.to_csv(output_path, index=False)

print(f"\nProcessed data saved to: {output_path}")