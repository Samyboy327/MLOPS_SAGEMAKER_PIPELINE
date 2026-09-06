import boto3
import sagemaker
import os

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")


REGION = "ap-south-1"

BUCKET = "rohit-telecom-churn-data-2026"

PIPELINE_NAME = "CustomerChurnPipeline"

ROLE_ARN = (
    "arn:aws:iam::419022575435:"
    "role/telecom-churn-sagemaker-role"
)

RAW_PROCESSED_DATA_URI = (
    f"s3://{BUCKET}/processed/customer_clean.csv"
)

TRAINING_IMAGE = (
    "720646828776.dkr.ecr.ap-south-1.amazonaws.com/"
    "sagemaker-scikit-learn:1.4-2-py312-cpu-py3"
)


boto_session = boto3.Session(
    region_name=REGION
)

sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    default_bucket=BUCKET
)


def get_pipeline():

    # --------------------------------------------------
    # 1. Processing
    # --------------------------------------------------

    processor = ScriptProcessor(
        image_uri=(
        "720646828776.dkr.ecr.ap-south-1.amazonaws.com/"
        "sagemaker-scikit-learn:1.4-2-cpu-py3"
        ),

        command=["python3"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=sagemaker_session
    )

    processing_step = ProcessingStep(
        name="DataProcessing",

        processor=processor,

        code=(
            "/var/lib/jenkins/workspace/"
            "MLOPS-SageMaker-Pipeline/"
            "src/processing/pipeline_processing.py"
        ),

        inputs=[
            ProcessingInput(
                source=RAW_PROCESSED_DATA_URI,
                destination="/opt/ml/processing/input"
            )
        ],

        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=(
                    f"s3://{BUCKET}/pipeline/processing-output/"
                )
            )
        ]
    )

    # --------------------------------------------------
    # 2. Training
    # --------------------------------------------------

    estimator = Estimator(
    image_uri=TRAINING_IMAGE,
    role=ROLE_ARN,
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"s3://{BUCKET}/model-artifacts/",
    base_job_name="telecom-churn-pipeline-training",
    entry_point="train.py",
    source_dir=TRAINING_DIR,
    sagemaker_session=sagemaker_session
    )



    training_step = TrainingStep(
        name="ModelTraining",

        estimator=estimator,

        inputs={
            "train": processing_step.properties.ProcessingOutputConfig.Outputs[
                "output-1"
            ].S3Output.S3Uri
        }
    )

    # --------------------------------------------------
    # 3. Pipeline
    # --------------------------------------------------

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[],
        steps=[
            processing_step,
            training_step
        ],
        sagemaker_session=sagemaker_session
    )

    return pipeline


if __name__ == "__main__":

    pipeline = get_pipeline()

    print(
        "SageMaker SDK version:",
        sagemaker.__version__
    )

    print(
        "Pipeline name:",
        PIPELINE_NAME
    )

    definition = pipeline.definition()

    print(
        "Pipeline definition generated successfully."
    )

    print(definition)

    print("Creating/updating SageMaker Pipeline...")

    pipeline.upsert(
        role_arn=ROLE_ARN
    )

    print("SageMaker Pipeline created/updated successfully.")