import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
)

print("📦 Reading eval metrics...", flush=True)
with open("reports_finetune/eval_finetune.json") as f:
    metrics = json.load(f)

acc = metrics["accuracy"]
if acc < 0.75:
    print(f"❌ Accuracy {acc} below threshold, not promoting", flush=True)
    exit(1)

print("📦 Reading run ID...", flush=True)
with open("finetune/run_id.txt") as f:
    run_id = f.read().strip()

print(f"🔍 Promoting run: {run_id}", flush=True)

client = MlflowClient()
versions = client.search_model_versions("name='MalariaClassifier'")

promoted = False
for v in versions:
    if v.run_id == run_id:
        client.transition_model_version_stage(
            name="MalariaClassifier",
            version=v.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"✅ Promoted version {v.version} to Production", flush=True)
        print(f"   Accuracy: {acc}", flush=True)
        print(f"   Run ID: {run_id}", flush=True)
        promoted = True
        break

if not promoted:
    print("❌ Could not find model version for run_id", flush=True)
    exit(1)