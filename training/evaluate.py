import os
import json
import joblib
import pandas as pd

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix
)

# --------------------------------------------------

# SageMaker directories

# --------------------------------------------------

MODEL_DIR = os.environ.get(
"SM_MODEL_DIR",
"/opt/ml/model"
)

TEST_DIR = os.environ.get(
"SM_CHANNEL_TEST",
"/opt/ml/input/data/test"
)

OUTPUT_DIR = os.environ.get(
"SM_OUTPUT_DATA_DIR",
"/opt/ml/output"
)

# --------------------------------------------------

# Locate model

# --------------------------------------------------

model_path = os.path.join(
MODEL_DIR,
"model.joblib"
)

print(f"Loading model from: {model_path}")

model = joblib.load(model_path)

print("Model loaded successfully.")

# --------------------------------------------------

# Locate evaluation data

# --------------------------------------------------

test_file = os.path.join(
TEST_DIR,
"test.csv"
)

print(f"Reading evaluation data from: {test_file}")

df = pd.read_csv(test_file)

print(f"Evaluation data shape: {df.shape}")

# --------------------------------------------------

# Prepare features and target

# --------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

# --------------------------------------------------

# Prediction

# --------------------------------------------------

print("Generating predictions...")

y_pred = model.predict(X)

# --------------------------------------------------

# Metrics

# --------------------------------------------------

accuracy = accuracy_score(y, y_pred)

precision = precision_score(
y,
y_pred,
pos_label="Yes"
)

recall = recall_score(
y,
y_pred,
pos_label="Yes"
)

f1 = f1_score(
y,
y_pred,
pos_label="Yes"
)

cm = confusion_matrix(
y,
y_pred,
labels=["No", "Yes"]
)

# --------------------------------------------------

# Print metrics

# --------------------------------------------------

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("Confusion Matrix:")
print(cm)

# --------------------------------------------------

# Save evaluation results

# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

evaluation = {
"metrics": {
"accuracy": float(accuracy),
"precision": float(precision),
"recall": float(recall),
"f1": float(f1)
}
}

evaluation_path = os.path.join(
OUTPUT_DIR,
"evaluation.json"
)

with open(evaluation_path, "w") as f:
    json.dump(evaluation, f, indent=2)

print(f"Evaluation results saved to: {evaluation_path}")
