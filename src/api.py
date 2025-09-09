from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import logging
from datetime import datetime
from config_loader import load_config, get_config_section
import os
from monitor import SimpleMonitor


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


class MLModelAPI:
    def __init__(self):
        self.config = self._load_configuration()
        self.logger = self._setup_logging()
        self.monitor = SimpleMonitor()
        self.app = FastAPI(title=self.config["api"]["title"])
        self.model = self._load_model()
        self._setup_routes()

    def _load_configuration(self):
        """Load and prepare configuration."""
        config = load_config()

        # Override MLflow URI if environment variable is set
        if "MLFLOW_TRACKING_URI" in os.environ:
            config["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]

        return {
            "api": get_config_section(config, "api"),
            "mlflow": get_config_section(config, "mlflow"),
            "data": get_config_section(config, "data"),
            "logging": get_config_section(config, "logging"),
        }

    def _setup_logging(self):
        """Setup logging configuration."""
        logging_config = self.config["logging"]
        logging.basicConfig(
            level=getattr(logging, logging_config["level"]),
            format=logging_config["format"],
        )
        return logging.getLogger(__name__)

    def _load_model(self):
        """Load model from MLflow registry."""
        try:
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])

            model_name = self.config["mlflow"]["model_name"]
            model_version = self.config["mlflow"]["model_version"]
            model_uri = f"models:/{model_name}/{model_version}"

            self.logger.info(f"Loading model from registry: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            self.logger.info("Model loaded successfully from registry")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None

    def _setup_routes(self):
        """Setup FastAPI routes."""
        self.app.get("/health")(self.health_check)
        self.app.get("/monitoring")(self.get_monitoring_stats)
        self.app.post("/predict", response_model=PredictionResponse)(self.predict)

    def health_check(self):
        """Health check endpoint."""
        if self.model is None:
            self.logger.error("Health check failed - model not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {"status": "healthy", "model_loaded": True}

    def get_monitoring_stats(self):
        """Simple monitoring dashboard endpoint."""
        stats = self.monitor.get_today_stats()
        return {
            "today_stats": stats,
            "monitoring_files": {
                "requests": str(self.monitor.requests_file),
                "failures": str(self.monitor.failures_file),
                "daily_stats": str(self.monitor.stats_file),
            },
            "message": "All requests and responses are automatically logged",
        }

    def _prepare_features(self, request: PredictionRequest):
        """Prepare input features for prediction."""
        return np.array(
            [
                [
                    request.sepal_length,
                    request.sepal_width,
                    request.petal_length,
                    request.petal_width,
                ]
            ]
        )

    def _make_prediction(self, features):
        """Make prediction and calculate confidence."""
        prediction = self.model.predict(features)[0]
        prediction_proba = self.model.predict_proba(features)[0]
        confidence = float(prediction_proba.max())

        # Map to class name using config
        class_names = self.config["data"]["class_names"]
        prediction_name = class_names[prediction]

        return int(prediction), prediction_name, confidence

    def _create_response_data(self, prediction, prediction_name, confidence):
        """Create response data dictionary."""
        return {
            "prediction": prediction,
            "prediction_name": prediction_name,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }

    def predict(self, request: PredictionRequest):
        """Prediction endpoint with integrated monitoring."""
        request_data = request.dict()

        if self.model is None:
            error_msg = "Model not loaded"
            self.logger.error(f"Prediction failed - {error_msg}")
            self.monitor.log_request(request_data, error=error_msg)
            raise HTTPException(status_code=503, detail=error_msg)

        try:
            # Prepare input and make prediction
            features = self._prepare_features(request)
            prediction, prediction_name, confidence = self._make_prediction(features)

            # Create response
            response_data = self._create_response_data(
                prediction, prediction_name, confidence
            )

            # Log successful request/response
            request_id = self.monitor.log_request(request_data, response_data)

            self.logger.info(
                f"Prediction successful [ID: {request_id}] - {prediction_name} (confidence: {confidence:.3f})"
            )

            return PredictionResponse(**response_data)

        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            self.logger.error(error_msg)
            self.monitor.log_request(request_data, error=error_msg)
            raise HTTPException(status_code=500, detail=str(e))

    def run(self):
        """Run the API server."""
        import uvicorn

        api_config = self.config["api"]
        uvicorn.run(self.app, host=api_config["host"], port=api_config["port"])


# Create global API instance
ml_api = MLModelAPI()
app = ml_api.app


if __name__ == "__main__":
    ml_api.run()
