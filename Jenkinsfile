pipeline {
    agent any

    environment {
        AWS_REGION = 'ap-south-1'
        SAGEMAKER_PIPELINE = 'CustomerChurnPipeline'
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Update SageMaker Pipeline') {
            steps {
                sh '''
                    /opt/sagemaker-pipeline-venv/bin/python pipeline/pipeline.py
                '''
            }
        }

        stage('Start SageMaker Pipeline') {
            steps {
                script {

                    def executionArn = sh(
                        script: """
                            aws sagemaker start-pipeline-execution \
                                --pipeline-name ${SAGEMAKER_PIPELINE} \
                                --region ${AWS_REGION} \
                                --query 'PipelineExecutionArn' \
                                --output text
                        """,
                        returnStdout: true
                    ).trim()

                    echo "SageMaker Pipeline Execution:"
                    echo executionArn

                    env.PIPELINE_EXECUTION_ARN = executionArn
                }
            }
        }

        stage('Wait for SageMaker Pipeline') {
            steps {
                script {

                    timeout(time: 30, unit: 'MINUTES') {

                        waitUntil {

                            def status = sh(
                                script: """
                                    aws sagemaker describe-pipeline-execution \
                                        --pipeline-execution-arn ${PIPELINE_EXECUTION_ARN} \
                                        --region ${AWS_REGION} \
                                        --query 'PipelineExecutionStatus' \
                                        --output text
                                """,
                                returnStdout: true
                            ).trim()

                            echo "SageMaker Pipeline Status: ${status}"

                            if (status == 'Succeeded') {
                                return true
                            }

                            if (status in ['Failed', 'Stopped']) {
                                error(
                                    "SageMaker Pipeline failed with status: ${status}"
                                )
                            }

                            sleep(time: 30, unit: 'SECONDS')

                            return false
                        }
                    }
                }
            }
        }

        stage('Find Latest Approved Model') {
            steps {
                script {

                    def modelPackageArn = sh(
                        script: """
                            aws sagemaker list-model-packages \
                                --model-package-group-name CustomerChurnModelGroup \
                                --region ${AWS_REGION} \
                                --sort-by CreationTime \
                                --sort-order Descending \
                                --max-results 20 \
                                --query 'ModelPackageSummaryList[?ModelApprovalStatus==`Approved`] | [0].ModelPackageArn' \
                                --output text
                        """,
                        returnStdout: true
                    ).trim()

                    if (!modelPackageArn || modelPackageArn == 'None') {
                        error('No Approved model package found.')
                    }

                    echo "Latest Approved Model Package:"
                    echo modelPackageArn

                    env.MODEL_PACKAGE_ARN = modelPackageArn
                }
            }
        }

                stage('Package Inference Source') {
            steps {
                sh '''
                    set -e

                    rm -rf /tmp/jenkins-churn-source
                    mkdir -p /tmp/jenkins-churn-source

                    cp training/inference.py /tmp/jenkins-churn-source/

                    tar -czf /tmp/jenkins-churn-source.tar.gz \
                        -C /tmp/jenkins-churn-source \
                        inference.py

                    S3_SOURCE_URI="s3://rohit-telecom-churn-data-2026/pipeline/inference-source/build-${BUILD_NUMBER}/sourcedir.tar.gz"

                    aws s3 cp \
                        /tmp/jenkins-churn-source.tar.gz \
                        "$S3_SOURCE_URI" \
                        --region "$AWS_REGION"

                    echo "Inference source uploaded:"
                    echo "$S3_SOURCE_URI"

                    echo "$S3_SOURCE_URI" > inference_source_uri.txt
                '''
            }
        }

        stage('Create SageMaker Model') {
            steps {
                script {

                    def modelName = "CustomerChurnModel-${env.BUILD_NUMBER}"

                    def containerInfo = sh(
                        script: """
                            aws sagemaker describe-model-package \
                                --model-package-name ${MODEL_PACKAGE_ARN} \
                                --region ${AWS_REGION} \
                                --query 'InferenceSpecification.Containers[0]' \
                                --output json
                        """,
                        returnStdout: true
                    ).trim()

                    writeFile(
                        file: 'container.json',
                        text: containerInfo
                    )

                    env.SAGEMAKER_MODEL_NAME = modelName

                    sh '''
                        set -e

                        IMAGE_URI=$(python3 -c "import json; print(json.load(open('container.json'))['Image'])")
                        MODEL_DATA_URL=$(python3 -c "import json; print(json.load(open('container.json'))['ModelDataUrl'])")

                        cat > primary-container.json <<EOF
{
    "Image": "$IMAGE_URI",
    "Mode": "SingleModel",
    "ModelDataUrl": "$MODEL_DATA_URL",
    "Environment": {
        "SAGEMAKER_PROGRAM": "inference.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "$(cat inference_source_uri.txt)",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        "SAGEMAKER_REGION": "ap-south-1"
    }
}
EOF

                        echo "Primary container configuration:"
                        cat primary-container.json

                        aws sagemaker create-model \
                            --model-name "$SAGEMAKER_MODEL_NAME" \
                            --execution-role-arn arn:aws:iam::419022575435:role/telecom-churn-sagemaker-role \
                            --primary-container file://primary-container.json \
                            --region "$AWS_REGION"
                    '''

                    echo "SageMaker Model Created:"
                    echo modelName
                }
            }
        }

        stage('Create Endpoint Configuration') {
            steps {
                script {

                    def endpointConfigName =
                        "customer-churn-endpoint-config-${env.BUILD_NUMBER}"

                    sh """
                        set -e

                        aws sagemaker create-endpoint-config \
                            --endpoint-config-name ${endpointConfigName} \
                            --production-variants \
                                VariantName=AllTraffic,ModelName=${SAGEMAKER_MODEL_NAME},InitialInstanceCount=1,InstanceType=ml.m5.large,InitialVariantWeight=1.0 \
                            --region ${AWS_REGION}
                    """

                    echo "SageMaker Endpoint Configuration Created:"
                    echo endpointConfigName

                    env.SAGEMAKER_ENDPOINT_CONFIG_NAME = endpointConfigName
                }
            }
        }

        stage('Update SageMaker Endpoint') {
            steps {
                script {

                    def endpointExists = sh(
                        script: """
                        aws sagemaker describe-endpoint \
                        --endpoint-name customer-churn-endpoint \
                        --region ${AWS_REGION} \
                        --query 'EndpointName' \
                        --output text 2>/dev/null || true
                """,
                    returnStdout: true
            ).trim()

            if (endpointExists == 'customer-churn-endpoint') {

                echo "Endpoint exists. Updating endpoint..."

                sh """
                    set -e

                    aws sagemaker update-endpoint \
                        --endpoint-name customer-churn-endpoint \
                        --endpoint-config-name ${SAGEMAKER_ENDPOINT_CONFIG_NAME} \
                        --region ${AWS_REGION}
                """

                echo "SageMaker Endpoint Update Started:"
                echo "customer-churn-endpoint"

            } else {

                echo "Endpoint does not exist. Creating endpoint..."

                sh """
                    set -e

                    aws sagemaker create-endpoint \
                        --endpoint-name customer-churn-endpoint \
                        --endpoint-config-name ${SAGEMAKER_ENDPOINT_CONFIG_NAME} \
                        --region ${AWS_REGION}
                """

                echo "SageMaker Endpoint Creation Started:"
                echo "customer-churn-endpoint"
            }
        }
    }
}


        
        stage('Wait for SageMaker Endpoint') {
            steps {
                script {

                    timeout(time: 20, unit: 'MINUTES') {

                        waitUntil {

                            def status = sh(
                                script: """
                                    aws sagemaker describe-endpoint \
                                        --endpoint-name customer-churn-endpoint \
                                        --region ${AWS_REGION} \
                                        --query 'EndpointStatus' \
                                        --output text
                                """,
                                returnStdout: true
                            ).trim()

                            echo "SageMaker Endpoint Status: ${status}"

                            if (status == 'InService') {
                                return true
                            }

                            if (status in ['Failed', 'OutOfService']) {
                                error(
                                    "SageMaker endpoint deployment failed with status: ${status}"
                                )
                            }

                            sleep(time: 30, unit: 'SECONDS')

                            return false
                        }
                    }
                }
            }
        }

        stage('Deployment Successful') {
            steps {
                echo 'SageMaker deployment completed successfully.'
            }
        }
    }
}

