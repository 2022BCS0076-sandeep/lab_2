pipeline {
    agent any

    environment {
        NAME   = "Sandeep"
        ROLLNO = "2022BCS0076"
        IMAGE  = "2022bcs0076-ml-model"
    }

    stages {

        stage("Checkout Code") {
            steps {
                checkout scm
            }
        }

        stage("Show Files") {
            steps {
                sh "ls -la"
            }
        }

        stage("Create Python venv + Install Dependencies") {
            steps {
                sh """
                python3 --version
                pip3 --version

                python3 -m venv venv
                . venv/bin/activate

                pip install --upgrade pip
                pip install -r requirements.txt
                """
            }
        }

        stage("Train + Evaluate Model") {
            steps {
                sh """
                . venv/bin/activate

                echo "======================================"
                echo "NAME   : ${NAME}"
                echo "ROLLNO : ${ROLLNO}"
                echo "======================================"

                python3 train.py
                """
            }
        }

        stage("Build Docker Image") {
            steps {
                sh """
                docker --version
                docker build -t ${IMAGE}:latest .
                docker images | head -20
                """
            }
        }
    }

    post {
        success {
            echo "✅ Pipeline SUCCESS: Training + Docker Build completed"
        }
        failure {
            echo "❌ Pipeline FAILED: Check Console Output"
        }
    }
}
