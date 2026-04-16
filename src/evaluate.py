import os
import yaml
import json
import torch
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from models import get_model


# ---------------- MLFLOW SETUP ---------------- #
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Malaria")

# ---------------- UTILS ---------------- #

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# ---------------- MAIN ---------------- #

def main():
    params = load_params()

    data_dir = params["data"]["resized_dir"]
    model_name = params["train"]["model"]
    batch_size = params["train"]["batch_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.ToTensor()
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ---------------- LOAD MODEL ---------------- #

    model = get_model(model_name, params)
    model.load_state_dict(torch.load(f"models/{model_name}.pth", map_location=device))
    model.to(device)
    model.eval()

    # ---------------- INFERENCE ---------------- #

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    metrics = compute_metrics(y_true, y_pred)

    # ---------------- CREATE DIRS ---------------- #

    os.makedirs("reports/plots", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)

    # ---------------- CONFUSION MATRIX ---------------- #

    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(cm)
    cm_df.index.name = "actual"
    cm_df.columns.name = "predicted"
    cm_df.to_csv("reports/plots/confusion_matrix.csv")

    # Image for MLflow
    cm_img_path = "reports/plots/confusion_matrix.png"
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(cm_img_path)
    plt.close()

    # ---------------- PER-CLASS F1 ---------------- #

    report = classification_report(y_true, y_pred, output_dict=True)

    per_class = []
    for cls, vals in report.items():
        if cls not in ["accuracy", "macro avg", "weighted avg"]:
            per_class.append({
                "class": cls,
                "f1_score": vals["f1-score"]
            })

    with open("reports/plots/per_class_f1.json", "w") as f:
        json.dump(per_class, f, indent=4)

    # ---------------- SCATTER ---------------- #

    scatter_data = [{
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"]
    }]

    with open("reports/plots/scatter_metrics.json", "w") as f:
        json.dump(scatter_data, f, indent=4)

    # ---------------- METRICS (RAW) ---------------- #

    metrics_path = "reports/metrics/eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # ---------------- METRICS (FOR PLOTS) ---------------- #
    # convert to list format required by DVC bar chart

    metrics_plot = []
    for k, v in metrics.items():
        metrics_plot.append({
            "metric": k,
            "value": v
        })

    with open("reports/plots/eval_metrics_plots.json", "w") as f:
        json.dump(metrics_plot, f, indent=4)

    # ---------------- MLFLOW LOGGING ---------------- #

    with mlflow.start_run(run_name="evaluation"):

        mlflow.log_metrics({
            "eval_accuracy": metrics["accuracy"],
            "eval_precision": metrics["precision"],
            "eval_recall": metrics["recall"],
            "eval_f1": metrics["f1"]
        })

        mlflow.log_param("model", model_name)
        mlflow.log_param("data_dir", data_dir)
        mlflow.log_param("batch_size", batch_size)

        mlflow.log_artifact(cm_img_path, artifact_path="plots")
        mlflow.log_artifact(metrics_path, artifact_path="metrics")  

    print("Evaluation complete:", metrics)


if __name__ == "__main__":
    main()