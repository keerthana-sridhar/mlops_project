import os
import json
import yaml
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import get_model

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

print("📦 Loading params...", flush=True)
params = load_params()
model_name = params["train"]["model"]
print(f"✅ Model name: {model_name}", flush=True)

DEVICE = "cpu"
MODEL_PATH = "finetune/checkpoint.pth"
DATA_DIR = "data/processed/resized/val"
OUTPUT = "reports_finetune/eval_finetune.json"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

os.makedirs("reports_finetune", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise Exception("No finetuned model found")

print(f"📂 Loading dataset from {DATA_DIR}...", flush=True)
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
print(f"✅ Dataset loaded: {len(dataset)} images, {len(loader)} batches", flush=True)

print("🧠 Loading model...", flush=True)
model = get_model(model_name, params)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
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
metrics = {"accuracy": round(acc, 4)}
print(f"✅ Evaluation complete | Final accuracy: {acc:.4f}", flush=True)

with open(OUTPUT, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"💾 Metrics saved to {OUTPUT}", flush=True)

if acc < 0.75:
    print(f"❌ Model rejected — accuracy {acc:.4f} below threshold 0.75", flush=True)
    exit(1)

print(f"✅ Model accepted — accuracy {acc:.4f}", flush=True)