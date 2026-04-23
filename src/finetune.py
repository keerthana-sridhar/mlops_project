import os
import yaml
import torch
import shutil
import datetime
import mlflow
import mlflow.pytorch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import get_model

DEVICE = "cpu"
PRODUCTION_CHECKPOINT = "finetune/checkpoint.pth"
CANDIDATE_CHECKPOINT = "finetune/candidate_checkpoint.pth"
RUN_ID_FILE = "finetune/run_id.txt"
DATA_DIR = "data/processed/incremental_resized"

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
)
mlflow.set_experiment("Malaria_Finetune")  # ← separate experiment name

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

print("📦 Loading params...", flush=True)
params = load_params()
model_name = params["train"]["model"]
lr = params["train"]["lr"]
print(f"✅ Model: {model_name}, LR: {lr}", flush=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

os.makedirs("finetune", exist_ok=True)

if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
    print("⚠️ No new data", flush=True)
    exit(0)

print(f"📂 Loading dataset from {DATA_DIR}...", flush=True)
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f"✅ Dataset loaded: {len(dataset)} images, {len(loader)} batches", flush=True)

model = get_model(model_name, params)
base_model_path = f"models/{model_name}.pth"

if os.path.exists(PRODUCTION_CHECKPOINT):
    model.load_state_dict(torch.load(PRODUCTION_CHECKPOINT, map_location=DEVICE))
    print("🔁 Loaded production checkpoint", flush=True)
elif os.path.exists(base_model_path):
    model.load_state_dict(torch.load(base_model_path, map_location=DEVICE))
    print(f"📦 Loaded base model: {base_model_path}", flush=True)
else:
    print("🆕 No checkpoint found, finetuning from random init", flush=True)

model.to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

run_name = f"finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    mlflow.log_params({
        "model": model_name,
        "type": "finetune",
        "epochs": 1,
        "lr": lr,
        "batch_size": 16,
        "num_new_images": len(dataset)
    })

    print("🏋️ Starting training...", flush=True)
    for epoch in range(1):
        total_loss = 0
        print(f"Epoch {epoch+1}", flush=True)
        for i, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 2 == 0:
                print(f"  Batch {i}/{len(loader)}, Loss {loss.item():.4f}", flush=True)

        avg_loss = total_loss / len(loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"✅ Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}", flush=True)

    # Save checkpoint
    torch.save(model.state_dict(), CANDIDATE_CHECKPOINT)
    print("💾 Candidate checkpoint saved", flush=True)

    # Log the candidate artifact to MLflow; only accepted runs are registered later.
    mlflow.log_artifact(CANDIDATE_CHECKPOINT, artifact_path="finetune_checkpoint")
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
    )
    mlflow.set_tag("candidate_checkpoint_path", CANDIDATE_CHECKPOINT)
    mlflow.set_tag("promotion_ready", "pending_evaluation")

    # Save run_id for promote step
    run_id = mlflow.active_run().info.run_id
    os.makedirs("finetune", exist_ok=True)
    with open(RUN_ID_FILE, "w") as f:
        f.write(run_id)
    print(f"✅ MLflow run logged: {run_id}", flush=True)

# Archive incremental data
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
archive_dir = f"data/processed/incremental_archive/{timestamp}"
shutil.copytree(DATA_DIR, archive_dir)
shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)
print(f"📦 Archived incremental data to {archive_dir}", flush=True)
print("✅ Finetuning complete", flush=True)
