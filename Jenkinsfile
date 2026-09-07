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

        stage('Pipeline Successful') {
            steps {
                echo 'SageMaker Pipeline completed successfully.'
            }
        }
    }
}