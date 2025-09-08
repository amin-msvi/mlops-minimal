# ML/Ops Engineer Assignment

You’ve received a basic ML model (see `train.py`) from a data scientist. Your task is to productionize this pipeline as described below.

Focus on structure, clarity, and real-world thinking.

You can use AI tools — but be ready to explain your decisions in the interview.

---

## Test Assignment

### Goal:
Simulate productionizing a simple ML model by designing a basic pipeline and infrastructure around it.

### Scenario:
Your team has developed a classification model. You’ve received a working `train.py` script from a data scientist. Now, you need to create an initial ML pipeline that allows for:

- Automated training
- Artifact tracking
- Containerized deployment of the trained model as a REST API
- Basic monitoring/logging of inference requests

### Your Task:

1. **Pipeline Design (Basic CI/CD + Reproducibility):**
   - Set up a simple ML pipeline with training and logging.
   - Use any orchestration approach (Airflow, Prefect, GitHub Actions, or shell script).
   - Track model training run, including hyperparameters and metrics (e.g., via MLflow or your preferred tool).

2. **Containerized Inference API:**
   - Package the trained model into a REST API using Flask, FastAPI, or similar.
   - Containerize it with Docker.

3. **Basic Monitoring (Simulated):**
   - Log incoming requests and responses with timestamps.
   - Simulate drift tracking or basic input stats collection in logs.

4. **README (Mandatory):**
   - Document:
     - How to run the training pipeline
     - How to build and run the inference container
     - What is monitored and how

---

## Submission Guidelines:

- Share a GitHub repo or archive with:
  - Source code
  - `Dockerfile` and `requirements.txt`
  - A `README.md` explaining your approach and trade-offs

---

## In the Interview, We'll Discuss:

- Why you chose the tools and structure
- How you’d extend the solution (e.g., model versioning, A/B testing)
- How you would scale or secure it in a production setting
- What parts you automated with AI and why
