import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from config_loader import load_config, get_config_section
import os
import logging


class IrisModelTrainer:
    def __init__(self):
        self.config = load_config()
        self.mlflow_config = get_config_section(self.config, "mlflow")
        self.model_config = get_config_section(self.config, "model")
        self.data_config = get_config_section(self.config, "data")
        self.training_config = get_config_section(self.config, "training")
        self.logger = self._setup_logging()
        # Override MLflow URI if environment variable is set
        if "MLFLOW_TRACKING_URI" in os.environ:
            self.logger.info(
                f"Overriding MLflow tracking URI: {os.environ['MLFLOW_TRACKING_URI']}"
            )
            self.mlflow_config["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]

        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _setup_logging(self):
        """Setup logging configuration."""
        logging_config = self.config["logging"]
        logging.basicConfig(
            level=getattr(logging, logging_config["level"]),
            format=logging_config["format"],
        )
        return logging.getLogger(__name__)

    def load_and_split_data(self):
        """Load iris dataset and split into train/test sets."""
        X, y = load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.data_config["test_size"]
        )

    def create_and_train_model(self):
        """Create and train the logistic regression model."""
        self.model = LogisticRegression(**self.model_config)
        self.model.fit(self.X_train, self.y_train)
        self.logger.info("Model training complete.")

    def evaluate_model(self):
        """Evaluate the trained model and return accuracy."""
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def setup_mlflow(self):
        """Set up MLflow tracking and experiment."""
        mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])
        mlflow.set_experiment(self.mlflow_config["experiment_name"])

    def log_model_to_mlflow(self, accuracy):
        """Log model, parameters, and metrics to MLflow."""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.model_config)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)

            # Create model signature
            signature = infer_signature(self.X_train, self.model.predict(self.X_train))

            # Log model
            model_info = mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path=self.training_config["artifact_path"],
                signature=signature,
                input_example=self.X_train,
                registered_model_name=self.mlflow_config["model_name"],
            )

            # Set tags
            mlflow.set_tags(self.training_config["tags"])

            self.logger.info(f"Model registered as: {self.mlflow_config['model_name']}")
            self.logger.info(f"Model URI: {model_info.model_uri}")

            return model_info

    def test_loaded_model(self, model_info):
        """Test the loaded model with sample predictions."""
        loaded_model = mlflow.sklearn.load_model(model_info.model_uri)
        predictions = loaded_model.predict(self.X_test)
        self.logger.info(f"Sample predictions: {predictions[:10]}")

    def train(self):
        """Main training pipeline."""
        self.load_and_split_data()
        self.create_and_train_model()
        accuracy = self.evaluate_model()
        self.setup_mlflow()
        model_info = self.log_model_to_mlflow(accuracy)
        self.test_loaded_model(model_info)


if __name__ == "__main__":
    trainer = IrisModelTrainer()
    trainer.train()
