import boto3
import sagemaker
import os

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAINING_DIR = os.path.join(
    PROJECT_ROOT,
    "training"
)


REGION = "ap-south-1"

BUCKET = "rohit-telecom-churn-data-2026"

PIPELINE_NAME = "CustomerChurnPipeline"

MODEL_PACKAGE_GROUP_NAME = "CustomerChurnModelGroup"

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
        output_path=(
            f"s3://{BUCKET}/model-artifacts/"
        ),
        base_job_name="telecom-churn-pipeline-training",
        entry_point="train.py",
        source_dir=TRAINING_DIR,
        sagemaker_session=sagemaker_session
    )


    training_step = TrainingStep(
        name="ModelTraining",

        estimator=estimator,

        inputs={
            "train": (
                processing_step
                .properties
                .ProcessingOutputConfig
                .Outputs["output-1"]
                .S3Output
                .S3Uri
            )
        }
    )


    # --------------------------------------------------
    # 3. Evaluation
    # --------------------------------------------------

    evaluation_processor = ScriptProcessor(
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


    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )


    evaluation_step = ProcessingStep(
        name="Evaluation",

        processor=evaluation_processor,

        code=(
            "/var/lib/jenkins/workspace/"
            "MLOPS-SageMaker-Pipeline/"
            "training/evaluate.py"
        ),

        inputs=[

            # Model artifact from Training
            ProcessingInput(
                source=(
                    training_step
                    .properties
                    .ModelArtifacts
                    .S3ModelArtifacts
                ),
                destination="/opt/ml/processing/model"
            ),

            # Processed dataset from Processing
            ProcessingInput(
                source=(
                    processing_step
                    .properties
                    .ProcessingOutputConfig
                    .Outputs["output-1"]
                    .S3Output
                    .S3Uri
                ),
                destination="/opt/ml/processing/data"
            )
        ],

        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=(
                    f"s3://{BUCKET}/pipeline/evaluation/"
                )
            )
        ],

        property_files=[
            evaluation_report
        ]
    )


    # --------------------------------------------------
    # 4. Model Registration
    # --------------------------------------------------

    register_model_step = RegisterModel(

        name="RegisterModel",

        estimator=estimator,

        model_data=(
            training_step
            .properties
            .ModelArtifacts
            .S3ModelArtifacts
        ),

        content_types=[
            "text/csv"
        ],

        response_types=[
            "text/csv"
        ],

        inference_instances=[
            "ml.m5.large"
        ],

        transform_instances=[
            "ml.m5.large"
        ],

        model_package_group_name=(
            MODEL_PACKAGE_GROUP_NAME
        ),

        approval_status=(
            "PendingManualApproval"
        )
    )


    # --------------------------------------------------
    # 5. Accuracy Check
    # --------------------------------------------------

    accuracy_condition = ConditionStep(

        name="AccuracyCheck",

        conditions=[

            ConditionGreaterThanOrEqualTo(

                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="metrics.accuracy.value"
                ),

                right=0.60
            )
        ],

        # If accuracy >= 60%
        if_steps=[
            register_model_step
        ],

        # If accuracy < 60%
        # Pipeline stops without registration
        else_steps=[]
    )


    # --------------------------------------------------
    # 6. Pipeline
    # --------------------------------------------------

    pipeline = Pipeline(

        name=PIPELINE_NAME,

        parameters=[],

        steps=[
            processing_step,
            training_step,
            evaluation_step,
            accuracy_condition
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

    print(
        "Creating/updating SageMaker Pipeline..."
    )

    pipeline.upsert(
        role_arn=ROLE_ARN
    )

    print(
        "SageMaker Pipeline created/updated successfully."
    )

