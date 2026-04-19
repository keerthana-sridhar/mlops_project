import os
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from models import get_model
import mlflow

import mlflow
import os

# Option 1: Use MLflow server (recommended)
#mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri(
    os.environ.get(
        "MLFLOW_TRACKING_URI", 
        "sqlite:///mlflow.db"   # relative = local project dir, works on terminal
    )
)
mlflow.set_experiment("Malaria")

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


# ---------------- DATA ---------------- #

def get_dataloaders(data_dir, batch_size):
    transform = transforms.ToTensor()

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
    )


# ---------------- METRICS ---------------- #

def compute_metrics(y_true, y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    return compute_metrics(y_true, y_pred)


# ---------------- TRAIN ---------------- #

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Batch {i} | Loss: {loss.item():.4f}")

    return total_loss / len(loader)


# ---------------- MAIN ---------------- #

def main():
    params = load_params()

    data_dir = params["data"]["resized_dir"]  # change to augmented_dir if needed
    batch_size = params["train"]["batch_size"]
    epochs = params["train"]["epochs"]
    lr = params["train"]["lr"]
    model_name = params["train"]["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)

    model = get_model(model_name, params).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 🔥 START MLFLOW RUN
    with mlflow.start_run():

        # ✅ Log parameters
        mlflow.log_params({
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "model": model_name
        })

        # ---------------- TRAIN LOOP ---------------- #
        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)

            print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f}")

            # ✅ Log per-epoch metrics
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_f1", val_metrics["f1"], step=epoch)

        # ---------------- TEST ---------------- #
        test_metrics = evaluate(model, test_loader, device)

        # ✅ Log final metrics
        mlflow.log_metrics({
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"]
        })

        # ---------------- SAVE MODEL ---------------- #
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}.pth"
        torch.save(model.state_dict(), model_path)

        # ✅ Log model file as artifact (SAFE way)
        mlflow.log_artifact(model_path, artifact_path="model_files")


        # 2. Log structured model (IMPORTANT)
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="MalariaClassifier"
)

        # ---------------- SAVE METRICS ---------------- #
        os.makedirs("reports/metrics", exist_ok=True)
        metrics_path = "reports/metrics/train_metrics.json"

        with open(metrics_path, "w") as f:
            json.dump({
                "val": val_metrics,
                "test": test_metrics
            }, f, indent=4)

        # ✅ Log metrics file
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

    print("Training complete. Metrics saved.")


if __name__ == "__main__":
    main()