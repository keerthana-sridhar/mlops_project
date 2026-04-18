
import io
import yaml
import logging 
import os
import json
import time
from fastapi import Request
import subprocess
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
import time
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

# ---------------- PROMETHEUS METRICS ---------------- #

# API metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"]
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "Request latency",
    ["endpoint"]
)

# Model metrics
predictions_total = Counter(
    "predictions_total",
    "Total predictions",
    ["label"]
)

prediction_confidence = Gauge(
    "prediction_confidence",
    "Latest confidence"
)

prediction_entropy = Gauge(
    "prediction_entropy",
    "Latest entropy"
)

low_confidence_total = Counter(
    "low_confidence_total",
    "Low confidence predictions"
)

inference_latency = Histogram(
    "inference_latency_seconds",
    "Inference latency"
)

# Drift
drift_detected = Gauge(
    "drift_detected",
    "1 if drift detected else 0"
)

image_mean_delta = Gauge(
    "image_mean_delta",
    "Difference from training mean"
)

# Pipeline / feedback
unlabelled_images_saved_total = Counter(
    "unlabelled_images_saved_total",
    "Images saved for feedback"
)

labelled_images_received_total = Counter(
    "labelled_images_received_total",
    "Labelled images uploaded"
)

retraining_triggered_total = Counter(
    "retraining_triggered_total",
    "Retraining runs triggered"
)

model_version = Gauge(
    "model_version",
    "Current model version"
)
invalid_image_uploads_total = Counter(
    "invalid_image_uploads_total",
    "Invalid uploads"
)
low_confidence_rate_gauge = Gauge(
    "low_confidence_rate",
    "Rolling low confidence rate"
)
failed_predictions_total = Counter(
    "failed_predictions_total",
    "Inference failures"
)
ood_total = Counter(
    "ood_predictions_total",
    "OOD images detected"
)

ood_rate_gauge = Gauge(
    "ood_rate",
    "Rolling OOD rate"
)

'''
# ---------------- METRICS STORAGE ---------------- #

METRICS_FILE = "inference_metrics.json"

if not os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "w") as f:
        json.dump({
            "total": 0,
            "low_conf": 0,
            "conf_sum": 0.0
        }, f)
'''
CONF_THRESHOLD = 0.6
DRIFT_THRESHOLD = 0.15
OOD_RATE_THRESHOLD = 0.1

_runtime_counts = {
    "total": 0,
    "low_conf": 0,
    "ood_count": 0,
    "conf_sum": 0.0
}

def update_runtime_counts(confidence, is_ood=False):
    _runtime_counts["total"] += 1
    _runtime_counts["conf_sum"] += confidence

    if confidence < CONF_THRESHOLD:
        _runtime_counts["low_conf"] += 1

    if is_ood:
        _runtime_counts["ood_count"] += 1


def compute_rates():
    total = _runtime_counts["total"]

    if total == 0:
        return 0.0, 0.0, 0.0

    low_conf_rate = _runtime_counts["low_conf"] / total
    ood_rate = _runtime_counts["ood_count"] / total
    avg_conf = _runtime_counts["conf_sum"] / total

    return low_conf_rate, ood_rate, avg_conf
'''

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
'''

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

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)



@app.get("/")
def root():
    return {"message": "API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"ready": True}


@app.middleware("http")
async def track_requests(request, call_next):
    start = time.time()

    response = await call_next(request)

    duration = time.time() - start

    http_requests_total.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()

    http_request_duration.labels(
        endpoint=request.url.path
    ).observe(duration)

    return response

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

        # -------- INPUT VALIDATION -------- #
        if file.content_type not in ["image/jpeg", "image/png"]:
            log_failure({
                "filename": file.filename,
                "type": "invalid_input",
                "reason": "unsupported_format",
                "timestamp": time.time()
            })
            invalid_image_uploads_total.inc()
            raise HTTPException(status_code=422, detail="Only JPG and PNG images are supported")

        image_bytes = await file.read()

        if not image_bytes:
            log_failure({
                "filename": file.filename,
                "type": "invalid_input",
                "reason": "empty_file",
                "timestamp": time.time()
            })
            invalid_image_uploads_total.inc()
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
            invalid_image_uploads_total.inc()
            raise HTTPException(status_code=400, detail="Invalid image file")

        # -------- PREPROCESS + DRIFT -------- #
        input_tensor = preprocess_image(image)

        mean_val = input_tensor.mean().item()
        baseline_mean = sum(BASELINE_MEAN) / len(BASELINE_MEAN)

        delta = abs(mean_val - baseline_mean)
        image_mean_delta.set(delta)

        is_ood = not is_valid_image(image, input_tensor)

        # -------- INFERENCE -------- #
        start_time = time.time()

        with torch.no_grad():
            outputs = model(input_tensor)

            probs = torch.softmax(outputs, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs.max().item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

        latency = time.time() - start_time
        inference_latency.observe(latency)

        # -------- POST-INFERENCE LOGIC -------- #
        label = class_names[pred]

        low_conf = confidence < CONF_THRESHOLD
        high_entropy = entropy > ENTROPY_THRESHOLD

        if is_ood:
            label = "ood"
            ood_total.inc()
        elif low_conf or high_entropy:
            label = "uncertain"

        # ---- CONFIDENCE VISIBILITY LOGIC ----
        if label in ["ood"]:
            display_confidence = None
        elif label in ["uncertain"]:
            display_confidence = confidence   # optional: you can keep or hide
        else:
            display_confidence = confidence

        # -------- MODEL METRICS -------- #
        predictions_total.labels(label=label).inc()

        prediction_confidence.set(confidence)
        prediction_entropy.set(entropy)

        if confidence < CONF_THRESHOLD:
            low_confidence_total.inc()

        '''# -------- RATE METRICS -------- #
        with open(METRICS_FILE, "r") as f:
            data = json.load(f)

        total = data["total"]
        low = data["low_conf"]

        low_conf_rate = (low / total) if total > 0 else 0
        low_confidence_rate_gauge.set(low_conf_rate)'''

        # ---- UPDATE RUNTIME COUNTS ----
        update_runtime_counts(
                0 if label in ["ood"] else confidence,
                is_ood
            )

        low_conf_rate, ood_rate, avg_conf = compute_rates()

        # ---- PUSH TO PROMETHEUS ----
        low_confidence_rate_gauge.set(low_conf_rate)
        ood_rate_gauge.set(ood_rate)

        # OOD rate
        '''try:
            ood_count = ood_total._value.get()
            total_pred = predictions_total._value.get()
            ood_rate = (ood_count / total_pred) if total_pred > 0 else 0
            ood_rate_gauge.set(ood_rate)
        except:
            pass'''

        # -------- DRIFT (COMBINED SIGNAL) -------- #
        if delta > 0.15 or low_conf_rate > DRIFT_THRESHOLD or ood_rate > OOD_RATE_THRESHOLD:
            drift_detected.set(1)
        else:
            drift_detected.set(0)

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
            filename = os.path.basename(file.filename)

            unique_name = f"{timestamp}_{int(time.time()*1000)}_{filename}"

            save_dir = UNLABELED_OOD_DIR if label in ["ood", "uncertain"] else UNLABELED_DIR

            path = os.path.join(save_dir, unique_name)

            with open(path, "wb") as f:
                f.write(image_bytes)

            logger.info(f"✅ Saved image to {path}")
            unlabelled_images_saved_total.inc()

        except Exception as e:
            logger.error(f"❌ Save failed: {e}")

        # -------- UPDATE JSON METRICS -------- #
        #update_metrics(confidence)

        return {
            "filename": file.filename,
            "prediction_label": label,
            "confidence": display_confidence,
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
        failed_predictions_total.inc()
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
# ---------------- METRICS ENDPOINT ---------------- #
'''
@app.get("/api/metrics")
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

'''
@app.get("/api/metrics")
def metrics():
    try:
        low_conf_rate, ood_rate, avg_conf = compute_rates()
        total = _runtime_counts["total"]

        return {
            "total_predictions": total,
            "average_confidence": round(avg_conf, 3),
            "low_confidence_rate": round(low_conf_rate, 3),
            "ood_rate": round(ood_rate, 3),
            "drift_detected": bool(drift_detected._value.get()),
            "model_version": int(model_version._value.get()) if model_version._value.get() else 0
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to compute metrics"
        )
    
@app.get("/api/failures")
def get_failures(limit: int = 50, type: str = None):
    try:
        with open(FAILURE_FILE, "r") as f:
            data = json.load(f)

        if type:
            data = [d for d in data if d.get("type") == type]

        return {
            "total": len(data),
            "entries": data[-limit:]
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to read failures"
        )
# ---------------- RETRAIN ---------------- #

@app.post("/retrain")
def retrain():
    try:
        subprocess.run(["dvc", "repro"])
        retraining_triggered_total.inc()
        return {"status": "Manual retraining triggered (Airflow not involved)"}
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
            model_version.set(int(version))
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
    

ACTIVE_ALERTS = []

@app.post("/internal/alert")
async def receive_alert(request: Request):
    body = await request.json()
    print("🚨 ALERT RECEIVED:", body)  # <-- ADD THIS
    global ACTIVE_ALERTS

    ACTIVE_ALERTS = [
        a for a in body.get("alerts", [])
        if a.get("status") == "firing"
    ]

    return {"received": True}


@app.get("/alerts")
def get_alerts():
    return {"alerts": ACTIVE_ALERTS}


