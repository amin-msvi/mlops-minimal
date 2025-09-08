from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = FastAPI(title="ML API")


def load_model():
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:8080")
        
        # Load model from registry
        model_name = "iris_classifier"
        model_version = "latest"  # or specify a version number
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

        # Map to class name
        class_names = ["Setosa", "Versicolor", "Virginica"]
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
    uvicorn.run(app, host="localhost", port=8000)
