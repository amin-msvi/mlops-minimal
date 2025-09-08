from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import logging
from datetime import datetime
from config_loader import load_config, get_config_section

# Load configuration
config = load_config()
api_config = get_config_section(config, "api")
mlflow_config = get_config_section(config, "mlflow")
data_config = get_config_section(config, "data")
logging_config = get_config_section(config, "logging")

# Setup logging
logging.basicConfig(
    level=getattr(logging, logging_config["level"]),
    format=logging_config["format"]
)
logger = logging.getLogger(__name__)

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
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully from registry")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

model = load_model()

# Req/res models
class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: int
    prediction_name: str
    timestamp: str


# Health check endpoint
@app.get("/health")
def health_check():
    if model is None:
        logger.error("Health check failed - model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        logger.error("Prediction failed - model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        features = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Map to class name using config
        class_names = data_config["class_names"]
        prediction_name = class_names[prediction]

        # Log the predictions
        logger.info(f"Prediction made: {prediction_name} for input {request.dict()}")

        return PredictionResponse(
            prediction=int(prediction),
            prediction_name=prediction_name,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=api_config["host"], port=api_config["port"])