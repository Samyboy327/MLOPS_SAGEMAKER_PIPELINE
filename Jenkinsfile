pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('SageMaker Training') {
            steps {
                sh '''
                    /opt/jenkins-mlops-venv/bin/python \
                    training/sagemaker_train.py
                '''
            }
        }
    }
}