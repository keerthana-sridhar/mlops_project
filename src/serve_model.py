import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name("Malaria")

if experiment is None:
    raise Exception("Experiment 'Malaria' not found")

experiment_id = experiment.experiment_id

# Get best run
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.test_f1 DESC"]
)

if not runs:
    raise Exception("No runs found")

best_run = runs[0]

run_id = best_run.info.run_id
run_name = best_run.data.tags.get("mlflow.runName", "unknown")

f1 = best_run.data.metrics.get("test_f1")
accuracy = best_run.data.metrics.get("test_accuracy")

# Find model version
model_name = "MalariaClassifier"

versions = client.search_model_versions(f"name='{model_name}'")

best_version = None
for v in versions:
    if v.run_id == run_id:
        best_version = v.version
        break

if best_version is None:
    raise Exception("No model version found")

# Promote
client.transition_model_version_stage(
    name=model_name,
    version=best_version,
    stage="Production",
    archive_existing_versions=True
)

# Pretty print info
print("\n🚀 MODEL PROMOTED TO PRODUCTION\n")
print(f"Model Name     : {model_name}")
print(f"Version        : {best_version}")
print(f"Run Name       : {run_name}")
print(f"Run ID         : {run_id}")
print(f"Test F1        : {f1}")
print(f"Test Accuracy  : {accuracy}")
print(f"Timestamp      : {datetime.now()}")
print("\n✅ Now serving this model via FastAPI\n")