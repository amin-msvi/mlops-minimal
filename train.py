import joblib
import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 42,
}

model = LogisticRegression(**params)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Iris Classification")

with mlflow.start_run():
    mlflow.log_params(params)

    mlflow.log_metric("accuracy", accuracy)

    signature = infer_signature(X_train, model.predict(X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="iris_classifier"
    )


    mlflow.set_tags({"Training Info": "Basic LogReg model for iris dataset"})

    print("model registered as: iris_classifier")
    print(f"Model URI: {model_info.model_uri}")


loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)
print(predictions[:10])
