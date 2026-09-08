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

                    sh """

                        set -e

                        IMAGE_URI=\$(python3 -c "import json; print(json.load(open('container.json'))['Image'])")
                        MODEL_DATA_URL=\$(python3 -c "import json; print(json.load(open('container.json'))['ModelDataUrl'])")

                        aws sagemaker create-model \
                            --model-name ${modelName} \
                            --execution-role-arn arn:aws:iam::419022575435:role/telecom-churn-sagemaker-role \
                            --primary-container Image=\$IMAGE_URI,ModelDataUrl=\$MODEL_DATA_URL \
                            --region ${AWS_REGION}
                    """

                    echo "SageMaker Model Created:"
                    echo modelName

                    env.SAGEMAKER_MODEL_NAME = modelName
                }
            }
        }

        stage('Pipeline Successful') {
            steps {
                echo 'SageMaker Pipeline completed successfully.'
            }
        }
    }
}