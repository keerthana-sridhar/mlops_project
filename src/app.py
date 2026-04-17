'''import io
import yaml
import logging

import subprocess

from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import datasets, transforms

import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


# ---------------- LOGGING ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------- LOAD CONFIG ---------------- #

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


params = load_params()
data_dir = params["data"]["resized_dir"]


# ---------------- LOAD CLASS NAMES ---------------- #

train_dataset = datasets.ImageFolder(f"{data_dir}/train")
class_names = train_dataset.classes


# ---------------- LOAD MODEL FROM MLFLOW ---------------- #

mlflow.set_tracking_uri("sqlite:///mlflow.db")

logger.info("Loading model from MLflow Production...")

model = mlflow.pytorch.load_model(
    "models:/MalariaClassifier/Production"
)

model.eval()


# ---------------- FETCH MODEL METADATA ---------------- #

client = MlflowClient()

latest_versions = client.get_latest_versions(
    name="MalariaClassifier",
    stages=["Production"]
)

if latest_versions:
    version_info = latest_versions[0]
    run_id = version_info.run_id
    version = version_info.version

    run = client.get_run(run_id)

    run_name = run.data.tags.get("mlflow.runName", "unknown")
    f1 = run.data.metrics.get("test_f1")
    acc = run.data.metrics.get("test_accuracy")

    logger.info("🚀 Serving MLflow Production Model")
    logger.info(f"Version: {version}")
    logger.info(f"Run Name: {run_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"Accuracy: {acc}")

else:
    logger.warning("No Production model found!")


# ---------------- PREPROCESS (MATCH TRAINING) ---------------- #

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # float32 + CHW
])


def preprocess_image(image):
    tensor = transform(image)          # (3, 224, 224)
    tensor = tensor.unsqueeze(0)       # (1, 3, 224, 224)
    return tensor


# ---------------- FASTAPI ---------------- #

app = FastAPI(title="Image Classification API")


@app.get("/")
def root():
    return {"message": "API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"ready": True}


# ---------------- PREDICT ---------------- #

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        logger.info(f"Received file: {file.filename}")

        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=422,
                detail="Only JPG and PNG images are supported"
            )

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file")

        try:
            input_tensor = preprocess_image(image)

            with torch.no_grad():
                outputs = model(input_tensor)
                pred = outputs.argmax(dim=1).item()
                confidence = 1.0

        except Exception as e:
            logger.error(f"Inference failed: {e}")

            # 👇 ADD THIS
            import traceback
            traceback.print_exc()

            raise HTTPException(
                status_code=500,
                detail=str(e)   # 👈 THIS IS KEY
            )
        label = class_names[pred]

        return {
            "filename": file.filename,
            "prediction_label": label,
            "confidence": confidence
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------- PIPELINE STATUS ---------------- #

@app.get("/pipeline/status")
def pipeline_status():
    try:
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True
        )

        dvc_output = result.stdout.strip()

        return {
            "dvc_status": dvc_output,
            "clean": "up to date" in dvc_output.lower()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch pipeline status"
        )

# ---------------- MODEL INFO ---------------- #

@app.get("/model/info")
def model_info():
    try:
        if latest_versions:
            return {
                "model_name": "MalariaClassifier",
                "version": version,
                "run_id": run_id,
                "run_name": run_name,
                "f1_score": f1,
                "accuracy": acc
            }

        return {"message": "No production model available"}

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch model info"
        )'''

import io
import yaml
import logging 
import os
import json
import time
import subprocess

from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import datasets, transforms

import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


# ---------------- LOGGING ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------- METRICS STORAGE ---------------- #

METRICS_FILE = "inference_metrics.json"

if not os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "w") as f:
        json.dump({
            "total": 0,
            "low_conf": 0,
            "conf_sum": 0.0
        }, f)

CONF_THRESHOLD = 0.6
DRIFT_THRESHOLD = 0.15


def update_metrics(confidence):
    try:
        with open(METRICS_FILE, "r") as f:
            data = json.load(f)

        data["total"] += 1
        data["conf_sum"] += confidence

        if confidence < CONF_THRESHOLD:
            data["low_conf"] += 1

        with open(METRICS_FILE, "w") as f:
            json.dump(data, f)

    except Exception as e:
        logger.error(f"Metrics update failed: {e}")


# ---------------- FEEDBACK STORAGE ---------------- #
FAILURE_FILE = "failure.json"

if not os.path.exists(FAILURE_FILE):
    with open(FAILURE_FILE, "w") as f:
        json.dump([], f)


def log_failure(entry):
    try:
        # Load existing data
        if os.path.exists(FAILURE_FILE):
            with open(FAILURE_FILE, "r") as f:
                data = json.load(f)
        else:
            data = []

        # Append new entry
        data.append(entry)

        # Write back
        with open(FAILURE_FILE, "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        logger.error(f"Failure logging failed: {e}")

UNLABELED_DIR = "data/feedback/unlabeled"
os.makedirs(UNLABELED_DIR, exist_ok=True)
UNLABELED_OOD_DIR = "data/feedback/unlabeled_ood"
os.makedirs(UNLABELED_OOD_DIR, exist_ok=True)


# ---------------- LOAD CONFIG ---------------- #

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


params = load_params()
data_dir = params["data"]["resized_dir"]

# ---------------- LOAD EDA METRICS ---------------- #

eda_metrics_path = os.path.join(
    params["eda_reports"]["metrics_dir"],
    "eda_metrics.json"
)

try:
    with open(eda_metrics_path, "r") as f:
        eda_metrics = json.load(f)
    logger.info(f"Loaded EDA metrics from {eda_metrics_path}")
except Exception as e:
    logger.error(f"Failed to load EDA metrics: {e}")
    eda_metrics = {}

# ---------------- EXTRACT BASELINES ---------------- #

image_size_stats = eda_metrics.get("image_size", {})
pixel_stats = eda_metrics.get("pixel_stats", {})

MIN_WIDTH = image_size_stats.get("min_width", 50)
MAX_WIDTH = image_size_stats.get("max_width", 300)
MIN_HEIGHT = image_size_stats.get("min_width", 50)
MAX_HEIGHT = image_size_stats.get("max_width", 300)

BASELINE_MEAN = pixel_stats.get("mean", [0.5, 0.5, 0.5])
BASELINE_STD = pixel_stats.get("std", [0.2, 0.2, 0.2])

MEAN_TOL = 0.2
STD_TOL = 0.2
ENTROPY_THRESHOLD = 0.5

# ---------------- LOAD CLASS NAMES ---------------- #

train_dataset = datasets.ImageFolder(f"{data_dir}/train")
class_names = train_dataset.classes


# ---------------- LOAD MODEL FROM MLFLOW ---------------- #

mlflow.set_tracking_uri("sqlite:///mlflow.db")

logger.info("Loading model from MLflow Production...")

model = mlflow.pytorch.load_model(
    "models:/MalariaClassifier/Production"
)

model.eval()


# ---------------- FETCH MODEL METADATA ---------------- #

client = MlflowClient()

latest_versions = client.get_latest_versions(
    name="MalariaClassifier",
    stages=["Production"]
)

if latest_versions:
    version_info = latest_versions[0]
    run_id = version_info.run_id
    version = version_info.version

    run = client.get_run(run_id)

    run_name = run.data.tags.get("mlflow.runName", "unknown")
    f1 = run.data.metrics.get("test_f1")
    acc = run.data.metrics.get("test_accuracy")

    logger.info("🚀 Serving MLflow Production Model")
    logger.info(f"Version: {version}")
    logger.info(f"Run Name: {run_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"Accuracy: {acc}")

else:
    logger.warning("No Production model found!")


# ---------------- PREPROCESS ---------------- #

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def preprocess_image(image):
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor


# ---------------- FASTAPI ---------------- #

app = FastAPI(title="Image Classification API")


@app.get("/")
def root():
    return {"message": "API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"ready": True}


# ---------------- PREDICT ---------------- #
# ---------------- OOD VALIDATION ---------------- #

def is_valid_image(image, tensor):
    width, height = image.size

    # Size check
    if width < MIN_WIDTH or width > MAX_WIDTH:
        return False

    if height < MIN_HEIGHT or height > MAX_HEIGHT:
        return False

    # Pixel stats check (per channel)
    mean = tensor.mean(dim=[0, 2, 3]).squeeze().tolist()
    std = tensor.std(dim=[0, 2, 3]).squeeze().tolist()

    for i in range(3):
        if abs(mean[i] - BASELINE_MEAN[i]) > MEAN_TOL:
            return False
        if abs(std[i] - BASELINE_STD[i]) > STD_TOL:
            return False

    return True


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        logger.info(f"Received file: {file.filename}")

        if file.content_type not in ["image/jpeg", "image/png"]:
            log_failure({
                "filename": file.filename,
                "type": "invalid_input",
                "reason": "unsupported_format",
                "timestamp": time.time()
            })
            raise HTTPException(
                status_code=422,
                detail="Only JPG and PNG images are supported"
    )
        image_bytes = await file.read()

        if not image_bytes:
            log_failure({
                "filename": file.filename,
                "type": "invalid_input",
                "reason": "empty_file",
                "timestamp": time.time()
            })
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            log_failure({
                "filename": file.filename,
                "type": "invalid_input",
                "reason": "corrupt_image",
                "timestamp": time.time()
            })
            raise HTTPException(status_code=400, detail="Invalid image file")

        try:
            input_tensor = preprocess_image(image)
            # -------- LAYER 1: INPUT VALIDATION -------- #
            '''if not is_valid_image(image, input_tensor):
                return {
                    "filename": file.filename,
                    "prediction_label": "invalid_input",
                    "confidence": 0.0,
                    "low_confidence": True,
                    "message": "Image does not resemble training data"
                }'''
            is_ood = not is_valid_image(image, input_tensor)

            with torch.no_grad():
                outputs = model(input_tensor)

                # ✅ FIXED CONFIDENCE
                probs = torch.softmax(outputs, dim=1)
                pred = probs.argmax(dim=1).item()
                confidence = probs.max().item()
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

        label = class_names[pred]

        low_conf = confidence < CONF_THRESHOLD
        high_entropy = entropy > ENTROPY_THRESHOLD

        if is_ood:
            label = "ood"
        elif low_conf or high_entropy:
            label = "uncertain"
        # -------- FAILURE LOGGING -------- #

        if label == "ood":
            log_failure({
                "filename": file.filename,
                "type": "ood",
                "reason": "distribution_mismatch",
                "timestamp": time.time()
            })

        elif label == "uncertain":
            log_failure({
                "filename": file.filename,
                "type": "uncertain",
                "confidence": confidence,
                "entropy": entropy,
                "timestamp": time.time()
            })

        # -------- SAVE IMAGE -------- #
        try:
            timestamp = int(time.time())
            filename = os.path.basename(file.filename)  # ✅ strips any folder path from ZIP
            unique_name = f"{timestamp}_{int(time.time()*1000)}_{filename}"
            #path = os.path.join(UNLABELED_DIR, unique_name)
            if label in ["ood", "uncertain"]:
                save_dir = UNLABELED_OOD_DIR
            else:
                save_dir = UNLABELED_DIR

            path = os.path.join(save_dir, unique_name)  # ✅ use unique_name, not raw file.filename
            with open(path, "wb") as f:
                f.write(image_bytes)
            logger.info(f"✅ Saved image to {path}")
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")

        # -------- UPDATE METRICS -------- #
        update_metrics(confidence)

        return {
    "filename": file.filename,
    "prediction_label": label,
    "confidence": confidence,
    "entropy": entropy,
    "low_confidence": low_conf,
    "high_entropy": high_entropy,
    "ood": is_ood,
    "message": (
        "Input deviates from training distribution"
        if is_ood else
        "Low confidence prediction"
        if label == "uncertain" else
        "Prediction successful"
    )
}

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------- METRICS ENDPOINT ---------------- #

@app.get("/metrics")
def metrics():
    try:
        with open(METRICS_FILE, "r") as f:
            data = json.load(f)

        total = data["total"]
        low_conf = data["low_conf"]
        conf_sum = data["conf_sum"]

        avg_conf = conf_sum / total if total else 0
        low_conf_rate = low_conf / total if total else 0
        drift = low_conf_rate > DRIFT_THRESHOLD if total else False

        return {
            "total_predictions": total,
            "average_confidence": avg_conf,
            "low_confidence_rate": low_conf_rate,
            "drift_detected": drift
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to compute metrics"
        )


# ---------------- RETRAIN ---------------- #

@app.post("/retrain")
def retrain():
    try:
        subprocess.run(["dvc", "repro"])
        return {"status": "Retraining triggered"}
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Retraining failed"
        )


# ---------------- PIPELINE STATUS ---------------- #

@app.get("/pipeline/status")
def pipeline_status():
    try:
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True
        )

        dvc_output = result.stdout.strip()

        return {
            "dvc_status": dvc_output,
            "clean": "up to date" in dvc_output.lower()
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch pipeline status"
        )


# ---------------- MODEL INFO ---------------- #

@app.get("/model/info")
def model_info():
    try:
        if latest_versions:
            return {
                "model_name": "MalariaClassifier",
                "version": version,
                "run_id": run_id,
                "run_name": run_name,
                "f1_score": f1,
                "accuracy": acc
            }

        return {"message": "No production model available"}

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch model info"
        )