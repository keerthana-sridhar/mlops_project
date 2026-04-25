import os
import json
import shutil
import time
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from mlflow_utils import configure_mlflow

configure_mlflow()

MODEL_NAME = "MalariaClassifier"
RUN_ID_FILE = "finetune/run_id.txt"
CANDIDATE_CHECKPOINT = "finetune/candidate_checkpoint.pth"
PRODUCTION_CHECKPOINT = "finetune/checkpoint.pth"

print("📦 Reading eval metrics...", flush=True)
with open("reports_finetune/eval_finetune.json") as f:
    metrics = json.load(f)

acc = metrics["accuracy"]
if acc < 0.75:
    print(f"❌ Accuracy {acc} below threshold, not promoting", flush=True)
    exit(1)

print("📦 Reading run ID...", flush=True)
with open(RUN_ID_FILE) as f:
    run_id = f.read().strip()

print(f"🔍 Promoting run: {run_id}", flush=True)

client = MlflowClient()
try:
    client.get_registered_model(MODEL_NAME)
except Exception:
    client.create_registered_model(MODEL_NAME)

model_uri = f"runs:/{run_id}/model"
registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
version = registered.version

for _ in range(30):
    current = client.get_model_version(name=MODEL_NAME, version=version)
    if current.status == "READY":
        break
    time.sleep(1)
else:
    print(f"❌ Timed out waiting for model version {version} registration", flush=True)
    exit(1)

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Production",
    archive_existing_versions=True
)

if not os.path.exists(CANDIDATE_CHECKPOINT):
    print("❌ Candidate checkpoint missing after successful evaluation", flush=True)
    exit(1)

shutil.copy2(CANDIDATE_CHECKPOINT, PRODUCTION_CHECKPOINT)
os.remove(CANDIDATE_CHECKPOINT)
if os.path.exists(RUN_ID_FILE):
    os.remove(RUN_ID_FILE)

print(f"✅ Promoted version {version} to Production", flush=True)
print(f"   Accuracy: {acc}", flush=True)
print(f"   Run ID: {run_id}", flush=True)
print("📦 Production checkpoint updated for serving and future warm-start finetuning", flush=True)
