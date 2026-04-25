import os
import json
import yaml
import torch
import mlflow
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import get_model
from mlflow_utils import configure_mlflow, log_reproducibility_tags

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

print("📦 Loading params...", flush=True)
params = load_params()
model_name = params["train"]["model"]
print(f"✅ Model name: {model_name}", flush=True)

DEVICE = "cpu"
CANDIDATE_MODEL_PATH = "finetune/candidate_checkpoint.pth"
RUN_ID_FILE = "finetune/run_id.txt"
DATA_DIR = "data/processed/resized/val"
OUTPUT = "reports_finetune/eval_finetune.json"
THRESHOLD = 0.75

configure_mlflow(MLFLOW_TRACKING_URI)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

os.makedirs("reports_finetune", exist_ok=True)

if not os.path.exists(CANDIDATE_MODEL_PATH):
    raise Exception("No finetuned model found")

print(f"📂 Loading dataset from {DATA_DIR}...", flush=True)
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
print(f"✅ Dataset loaded: {len(dataset)} images, {len(loader)} batches", flush=True)

print("🧠 Loading model...", flush=True)
model = get_model(model_name, params)
model.load_state_dict(torch.load(CANDIDATE_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ Model loaded and set to eval mode", flush=True)

print("🔍 Running evaluation...", flush=True)
correct, total = 0, 0

with torch.no_grad():
    for i, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = torch.argmax(model(x), dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        if i % 5 == 0:
            print(f"  Batch {i}/{len(loader)} | Running acc: {correct/total:.4f}", flush=True)

acc = correct / total if total else 0
accepted = acc >= THRESHOLD
metrics = {
    "accuracy": round(acc, 4),
    "threshold": THRESHOLD,
    "accepted": accepted,
}
print(f"✅ Evaluation complete | Final accuracy: {acc:.4f}", flush=True)

with open(OUTPUT, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"💾 Metrics saved to {OUTPUT}", flush=True)

run_id = None
if os.path.exists(RUN_ID_FILE):
    with open(RUN_ID_FILE) as f:
        run_id = f.read().strip()

if run_id:
    with mlflow.start_run(run_id=run_id):
        log_reproducibility_tags({"pipeline_role": "airflow_finetune_evaluate"})
        mlflow.log_metric("finetune_validation_accuracy", acc)
        mlflow.log_metric("promotion_threshold", THRESHOLD)
        mlflow.log_artifact(OUTPUT, artifact_path="evaluation")
        mlflow.set_tag("promotion_ready", str(accepted).lower())
        mlflow.set_tag("finetune_candidate_status", "accepted" if accepted else "rejected")
        if not accepted:
            mlflow.set_tag("rejection_reason", f"accuracy_below_{THRESHOLD}")

if not accepted:
    if os.path.exists(CANDIDATE_MODEL_PATH):
        os.remove(CANDIDATE_MODEL_PATH)
    if os.path.exists(RUN_ID_FILE):
        os.remove(RUN_ID_FILE)
    print(f"❌ Model rejected — accuracy {acc:.4f} below threshold {THRESHOLD}", flush=True)
    print("🧹 Rejected candidate checkpoint removed to keep DVC clean", flush=True)
    exit(1)

print(f"✅ Model accepted — accuracy {acc:.4f}", flush=True)
