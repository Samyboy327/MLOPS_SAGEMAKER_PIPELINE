import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

TRAIN_DIR = os.environ.get(
    "SM_CHANNEL_TRAIN",
    r"D:\MLOPS_S3_DATA_PIPELINE\training\data"
)

train_file = os.path.join(TRAIN_DIR, "customer_clean.csv")

print(f"Reading training data from: {train_file}")

df = pd.read_csv(train_file)

print(f"Training data shape: {df.shape}")
print(df.head())

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("Numerical columns:")
print(X.select_dtypes(include=["int64", "float64"]).columns.tolist())

print("Categorical columns:")
print(X.select_dtypes(include=["object"]).columns.tolist())

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

print("Preprocessor created successfully.")

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

print("Model pipeline created successfully.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save test data for SageMaker evaluation
test_dir = os.environ.get(
    "SM_OUTPUT_DATA_DIR",
    "/opt/ml/output"
)

os.makedirs(test_dir, exist_ok=True)

test_data = X_test.copy()
test_data["Churn"] = y_test

test_path = os.path.join(
    test_dir,
    "test.csv"
)

test_data.to_csv(test_path, index=False)

print(f"Test data saved to: {test_path}")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("Starting model training...")

model.fit(X_train, y_train)

print("Model training completed.")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")

fitted_preprocessor = model.named_steps["preprocessor"]

X_test_transformed = fitted_preprocessor.transform(X_test)

print("Original test shape:", X_test.shape)
print("Transformed test shape:", X_test_transformed.shape)

precision = precision_score(y_test, y_pred, pos_label="Yes")
recall = recall_score(y_test, y_pred, pos_label="Yes")
f1 = f1_score(y_test, y_pred, pos_label="Yes")
cm = confusion_matrix(y_test, y_pred, labels=["No", "Yes"])

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("Confusion Matrix:")
print(cm)

model_dir = os.environ.get("SM_MODEL_DIR", "./model")

os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.joblib")

joblib.dump(model, model_path)

print(f"Model saved to: {model_path}")