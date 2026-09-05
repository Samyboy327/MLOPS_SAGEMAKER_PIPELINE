import boto3
import sagemaker

from sagemaker.workflow.pipeline import Pipeline


REGION = "ap-south-1"

BUCKET = "rohit-telecom-churn-data-2026"

PIPELINE_NAME = "CustomerChurnPipeline"

ROLE_ARN = (
    "arn:aws:iam::419022575435:"
    "role/telecom-churn-sagemaker-role"
)


boto_session = boto3.Session(region_name=REGION)

sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    default_bucket=BUCKET
)


def get_pipeline():
    """
    Create and return the SageMaker Pipeline definition.
    """

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session
    )

    return pipeline


if __name__ == "__main__":

    pipeline = get_pipeline()

    print("SageMaker SDK version:", sagemaker.__version__)
    print("Pipeline name:", PIPELINE_NAME)

    definition = pipeline.definition()

    print("Pipeline definition generated successfully.")
    print(definition)