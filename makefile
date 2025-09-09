PYTHON = python3

# Install dependencies
install:
	pip install -r requirements.txt

# Run ruff linting and formatting
lint:
	ruff check .
	ruff format --check .

# Format code with ruff
format:
	ruff format .

# Start MLflow server
mlflow:
	mkdir -p mlruns mlartifacts
	mlflow server --host 0.0.0.0 --port 8080 \
		--backend-store-uri file://$(PWD)/mlruns \
		--default-artifact-root file://$(PWD)/mlartifacts

# Train the model
train:
	MLFLOW_TRACKING_URI=http://localhost:8080 $(PYTHON) src/train.py

# Start the API server
api:
	MLFLOW_TRACKING_URI=http://localhost:8080 $(PYTHON) src/api.py

# Clean up
clean:
	pkill -f "mlflow server" || true
	pkill -f "uvicorn" || true
	rm -rf mlruns mlartifacts

# Run all services in sequence
run-all:
	@make mlflow &
	@sleep 10
	@make train
	@make api

.PHONY: install lint format mlflow train api clean run-all
