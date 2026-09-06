import os
import json
import tarfile
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------------------------
# Directories
# --------------------------------------------------

MODEL_INPUT_DIR = "/opt/ml/processing/model"
DATA_INPUT_DIR = "/opt/ml/processing/data"
OUTPUT_DIR = "/opt/ml/processing/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting model evaluation...")


# --------------------------------------------------
# Locate model.tar.gz
# --------------------------------------------------

model_tar = None

for file_name in os.listdir(MODEL_INPUT_DIR):
    if file_name.endswith(".tar.gz"):
        model_tar = os.path.join(
            MODEL_INPUT_DIR,
            file_name
        )
        break

if model_tar is None:
    raise FileNotFoundError(
        f"No model.tar.gz found in {MODEL_INPUT_DIR}"
    )

print(f"Model artifact found: {model_tar}")


# --------------------------------------------------
# Extract model
# --------------------------------------------------

extract_dir = "/opt/ml/processing/extracted_model"

os.makedirs(
    extract_dir,
    exist_ok=True
)

with tarfile.open(model_tar, "r:gz") as tar:
    tar.extractall(path=extract_dir)

model_path = os.path.join(
    extract_dir,
    "model.joblib"
)

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"model.joblib not found at {model_path}"
    )

model = joblib.load(model_path)

print("Model loaded successfully.")


# --------------------------------------------------
# Locate processed dataset
# --------------------------------------------------

data_file = os.path.join(
    DATA_INPUT_DIR,
    "customer_clean.csv"
)

if not os.path.exists(data_file):
    raise FileNotFoundError(
        f"Processed dataset not found at {data_file}"
    )

print(f"Reading evaluation data from: {data_file}")

df = pd.read_csv(data_file)

print(f"Evaluation dataset shape: {df.shape}")


# --------------------------------------------------
# Recreate the same test split used during training
# --------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Evaluation test data shape: {X_test.shape}")


# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

print("Predictions generated successfully.")


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

# Probability for ROC AUC
y_probability = model.predict_proba(X_test)[:, 1]

y_test_binary = (
    y_test == "Yes"
).astype(int)

roc_auc = roc_auc_score(
    y_test_binary,
    y_probability
)


print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")


# --------------------------------------------------
# Evaluation report
# --------------------------------------------------

evaluation_report = {
    "metrics": {
        "accuracy": {
            "value": float(accuracy)
        },
        "precision": {
            "value": float(precision)
        },
        "recall": {
            "value": float(recall)
        },
        "f1": {
            "value": float(f1)
        },
        "roc_auc": {
            "value": float(roc_auc)
        }
    }
}


evaluation_path = os.path.join(
    OUTPUT_DIR,
    "evaluation.json"
)

with open(
    evaluation_path,
    "w"
) as f:
    json.dump(
        evaluation_report,
        f,
        indent=4
    )

print(
    f"Evaluation report saved to: {evaluation_path}"
)

print("Model evaluation completed successfully.")

