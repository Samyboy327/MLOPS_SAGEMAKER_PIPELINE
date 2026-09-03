import boto3

from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, SourceCode, InputData
from sagemaker.core.shapes import OutputDataConfig


# --------------------------------------------------
# 1. AWS Session
# --------------------------------------------------
'''
boto_session = boto3.Session(
    profile_name="default",
    region_name="ap-south-1"
)

print("AWS Region:", boto_session.region_name)
'''

# --------------------------------------------------
# 2. SageMaker Execution Role
# --------------------------------------------------

ROLE_ARN = (
    "arn:aws:iam::419022575435:role/"
    "telecom-churn-sagemaker-role"
)

print("SageMaker Role:", ROLE_ARN)


# --------------------------------------------------
# 3. SageMaker Training Image
# --------------------------------------------------

TRAINING_IMAGE = (
    "720646828776.dkr.ecr.ap-south-1.amazonaws.com/"
    "sagemaker-scikit-learn:1.4-2-py312-cpu-py3"
)

print("Training image:", TRAINING_IMAGE)


# --------------------------------------------------
# 4. Source Code Configuration
# --------------------------------------------------

source_code = SourceCode(
    source_dir="training",
    entry_script="train.py"
)

print("SourceCode configured successfully")


# --------------------------------------------------
# 5. Compute Configuration
# --------------------------------------------------

compute = Compute(
    instance_type="ml.m5.large",
    instance_count=1
)

print("Compute configuration created")


# --------------------------------------------------
# 6. Training Data Configuration
# --------------------------------------------------

training_data = InputData(
    channel_name="train",
    data_source=(
        "s3://rohit-telecom-churn-data-2026/"
        "processed/customer_clean.csv"
    ),
    content_type="text/csv"
)

print("Training data configured successfully")


# --------------------------------------------------
# 7. Output Model Configuration
# --------------------------------------------------

output_config = OutputDataConfig(
    s3_output_path=(
        "s3://rohit-telecom-churn-data-2026/"
        "model-artifacts/"
    )
)

print("Output configuration created")


# --------------------------------------------------
# 8. Create ModelTrainer
# --------------------------------------------------

trainer = ModelTrainer(
    training_image=TRAINING_IMAGE,
    source_code=source_code,
    role=ROLE_ARN,
    compute=compute,
    output_data_config=output_config,
    base_job_name="telecom-churn-training"
)

print("ModelTrainer configured successfully")


# --------------------------------------------------
# 9. START SAGEMAKER TRAINING JOB
# --------------------------------------------------

print("\nStarting SageMaker training job...\n")

trainer.train(
    input_data_config=[training_data],
    wait=True,
    logs=True
)

print("\nSageMaker training job completed successfully.")