import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from config_loader import load_config, get_config_section

# Load configuration
config = load_config()
mlflow_config = get_config_section(config, "mlflow")
model_config = get_config_section(config, "model")
data_config = get_config_section(config, "data")
training_config = get_config_section(config, "training")

# Load and split data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=data_config["test_size"]
)

# Create model with parameters from config
model = LogisticRegression(**model_config)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Set up MLflow
mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
mlflow.set_experiment(mlflow_config["experiment_name"])

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(model_config)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Create model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=training_config["artifact_path"],
        signature=signature,
        input_example=X_train,
        registered_model_name=mlflow_config["model_name"],
    )

    # Set tags
    mlflow.set_tags(training_config["tags"])

    print(f"Model registered as: {mlflow_config['model_name']}")
    print(f"Model URI: {model_info.model_uri}")

# Test loaded model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)
print(f"Sample predictions: {predictions[:10]}")
