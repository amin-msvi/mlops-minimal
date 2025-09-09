# MLOps Minimal

A simple ML pipeline that trains an Iris classification model and serves it via a REST API with MLflow tracking.

## What it does

- Trains a logistic regression model on the Iris dataset
- Tracks experiments and models using MLflow
- Serves the trained model via a FastAPI REST API
- Logs requests and responses for basic monitoring

## Quick Start

### Option 1: Docker Compose (Recommended)
```bash
# Start everything
docker compose up -d

# Wait ~3 minutes for training to complete, then test
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# View MLflow UI
open http://localhost:8080

# Stop services
docker compose down
```

### Option 2: Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the required services by one command
make run-all

# If you want to run services separately, check out the makefile commands.

# Test API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## Endpoints

- `GET /health` - API health check
- `POST /predict` - Make predictions
- `GET /monitoring` - View request statistics
- MLflow UI: `http://localhost:8080`

## Monitoring

Request logs and statistics are automatically saved in the `monitoring_logs/` directory.

## Configuration

Edit `config.yaml` to modify model parameters, API settings, or MLflow configuration.

## How I could make it better
Since the purpose of this repo is to be as minimal as possible, and I was told to avoid adding unnecessary features, I have not added the following features that would be useful in a production-ready MLOps pipeline:
- Add unit and integration tests
- Using more advanced ML pipelines. For instance, Kubeflow pipelines:
    - Separating data preprocessing, training, evaluation, and deployment into distinct steps
    - Every step then could utilize different resources. For instance, based on the needs, it could use distributed data processing frameworks like Spark. Also, using GPU-enabled nodes for training deep learning models.
    - Recurring pipelines for periodic retraining and evaluation
- More advanced data/model drift detection, which could me implemented manually or using tools like EvidentlyAI
- Using cloud storage for artifacts (e.g., S3, GCS)
- Using feature stores for managing and serving features (e.g., Feast)
- Implementing data versioning (e.g., DVC)
- Implement more advanced monitoring and alerting (e.g., Prometheus, Grafana)

