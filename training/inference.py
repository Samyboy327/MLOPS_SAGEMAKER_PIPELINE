import os
import json
import joblib
import pandas as pd


def model_fn(model_dir):
    """
    Load the trained model from SageMaker's model directory.
    """

    model_path = os.path.join(model_dir, "model.joblib")

    model = joblib.load(model_path)

    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize the incoming request into a pandas DataFrame.
    """

    if request_content_type == "application/json":

        data = json.loads(request_body)

        if "features" in data:
            data = data["features"]

        if isinstance(data, dict):
            data = [data]

        return pd.DataFrame(data)

    raise ValueError(
        f"Unsupported content type: {request_content_type}"
    )


def predict_fn(input_data, model):
    """
    Generate prediction and churn probability.
    """

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    results = []

    for i in range(len(prediction)):
        results.append({
            "prediction": prediction[i],
            "churn_probability": float(probability[i][1])
        })

    return results


def output_fn(prediction, accept):
    """
    Serialize the prediction response.
    """

    if accept == "application/json":

        return json.dumps({
            "predictions": prediction
        }), accept

    raise ValueError(
        f"Unsupported accept type: {accept}"
    )