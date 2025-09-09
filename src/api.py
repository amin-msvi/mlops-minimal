from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import logging
from datetime import datetime
from config_loader import load_config, get_config_section
import os
from monitor import SimpleMonitor


# Load configuration
config = load_config()
api_config = get_config_section(config, "api")
mlflow_config = get_config_section(config, "mlflow")
data_config = get_config_section(config, "data")
logging_config = get_config_section(config, "logging")

if "MLFLOW_TRACKING_URI" in os.environ:
    mlflow_config["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]

# Setup logging
logging.basicConfig(
    level=getattr(logging, logging_config["level"]), format=logging_config["format"]
)
logger = logging.getLogger(__name__)

monitor = SimpleMonitor()

app = FastAPI(title=api_config["title"])


def load_model():
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

        # Load model from registry
        model_name = mlflow_config["model_name"]
        model_version = mlflow_config["model_version"]
        model_uri = f"models:/{model_name}/{model_version}"

        logger.info(f"Loading model from registry: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully from registry")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


model = load_model()


# Request/Response models
class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionResponse(BaseModel):
    prediction: int
    prediction_name: str
    confidence: float
    timestamp: str


# Health check endpoint
@app.get("/health")
def health_check():
    if model is None:
        logger.error("Health check failed - model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


# Monitoring dashboard endpoint
@app.get("/monitoring")
def get_monitoring_stats():
    """Simple monitoring dashboard"""
    stats = monitor.get_today_stats()
    return {
        "today_stats": stats,
        "monitoring_files": {
            "requests": str(monitor.requests_file),
            "failures": str(monitor.failures_file),
            "daily_stats": str(monitor.stats_file)
        },
        "message": "All requests and responses are automatically logged"
    }


# Prediction endpoint with integrated monitoring
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    request_data = request.dict()
    
    if model is None:
        error_msg = "Model not loaded"
        logger.error(f"Prediction failed - {error_msg}")
        
        # Log failed request
        monitor.log_request(request_data, error=error_msg)
        
        raise HTTPException(status_code=503, detail=error_msg)

    try:
        # Prepare input
        features = np.array(
            [
                [
                    request.sepal_length,
                    request.sepal_width,
                    request.petal_length,
                    request.petal_width,
                ]
            ]
        )

        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        confidence = float(prediction_proba.max())

        # Map to class name using config
        class_names = data_config["class_names"]
        prediction_name = class_names[prediction]

        # Create response
        response_data = {
            "prediction": int(prediction),
            "prediction_name": prediction_name,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }

        # Log successful request/response - THIS IS THE KEY PART
        request_id = monitor.log_request(request_data, response_data)
        
        logger.info(f"Prediction successful [ID: {request_id}] - {prediction_name} (confidence: {confidence:.3f})")

        return PredictionResponse(**response_data)

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(error_msg)
        
        # Log failed request - THIS CAPTURES FAILURES
        monitor.log_request(request_data, error=error_msg)
        
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=api_config["host"], port=api_config["port"])
