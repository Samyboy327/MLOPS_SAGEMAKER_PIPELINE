import os
import json
import logging
import joblib
import pandas as pd


# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """
    Load the trained model from SageMaker's model directory.
    """

    model_path = os.path.join(model_dir, "model.joblib")

    logger.info("Loading model from: %s", model_path)

    model = joblib.load(model_path)

    logger.info("Model loaded successfully.")

    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize the incoming request into a pandas DataFrame.
    """

    logger.info(
        "Received inference request. Content-Type: %s",
        request_content_type
    )

    if request_content_type == "application/json":

        data = json.loads(request_body)

        if "features" in data:
            data = data["features"]

        if isinstance(data, dict):
            data = [data]

        input_data = pd.DataFrame(data)

        logger.info(
            "Inference input processed successfully. Rows: %d, Columns: %d",
            input_data.shape[0],
            input_data.shape[1]
        )

        return input_data

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

        churn_probability = float(probability[i][1])

        result = {
            "prediction": prediction[i],
            "churn_probability": churn_probability
        }

        results.append(result)

        logger.info(
            "Prediction generated: prediction=%s, churn_probability=%.4f",
            prediction[i],
            churn_probability
        )

    logger.info(
        "Inference completed successfully. Predictions generated: %d",
        len(results)
    )

    return results


def output_fn(prediction, accept):
    """
    Serialize the prediction response.
    """

    if accept == "application/json":

        response = {
            "predictions": prediction
        }

        logger.info("Returning inference response.")

        return json.dumps(response), accept

    raise ValueError(
        f"Unsupported accept type: {accept}"
    )
