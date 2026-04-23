
import io
import yaml
import logging 
import os
import psutil   
import json
import time
import base64
import threading
import re
import zipfile
from fastapi import Request
import subprocess
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
import time
from PIL import Image, UnidentifiedImageError
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
from collections import deque

sys.path.append("/opt/project/src")


# 1. Add src to path
sys.path.insert(0, "/opt/project/src")

# 2. Import your module
import models

# 3. Force mapping (THIS is key)
sys.modules["models"] = models

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
confidence_sum = Counter(
    "prediction_confidence_sum",
    "Sum of prediction confidences"
)

confidence_count = Counter(
    "prediction_confidence_count",
    "Number of predictions"
)
cpu_usage = Gauge(
    "cpu_usage_percent",
    "CPU usage percentage"
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
    "ood_count": 0
}
_global_counts = {
    "invalid": 0,
    "ood": 0
}
_confidence_stats = {
    "sum": 0.0,
    "count": 0
}

def update_runtime_counts(confidence, is_ood=False):
    _runtime_counts["total"] += 1
    _confidence_stats["sum"] += confidence
    _confidence_stats["count"] += 1

    if confidence < CONF_THRESHOLD:
        _runtime_counts["low_conf"] += 1

    if is_ood:
        _runtime_counts["ood_count"] += 1


def compute_rates():
    total = _runtime_counts["total"]
    conf_count = _confidence_stats["count"]

    if total == 0:
        return 0.0, 0.0, 0.0

    low_conf_rate = _runtime_counts["low_conf"] / total
    ood_rate = _runtime_counts["ood_count"] / total
    avg_conf = (_confidence_stats["sum"] / conf_count) if conf_count else 0.0

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
FEEDBACK_ROOT = Path("data/feedback")
LABELLED_DIR = FEEDBACK_ROOT / "labelled"
for class_name in ["Parasitized", "Uninfected"]:
    (LABELLED_DIR / class_name).mkdir(parents=True, exist_ok=True)

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

UNLABELED_DIR = str(FEEDBACK_ROOT / "unlabeled")
os.makedirs(UNLABELED_DIR, exist_ok=True)
UNLABELED_OOD_DIR = str(FEEDBACK_ROOT / "unlabeled_ood")
os.makedirs(UNLABELED_OOD_DIR, exist_ok=True)

AIRFLOW_DAG_ID = os.environ.get("AIRFLOW_DAG_ID", "malaria_retraining_v2")
AIRFLOW_API_USERNAME = os.environ.get("AIRFLOW_API_USERNAME", "airflow")
AIRFLOW_API_PASSWORD = os.environ.get("AIRFLOW_API_PASSWORD", "airflow")
_AIRFLOW_API_HINT = os.environ.get("AIRFLOW_API_BASE", "").rstrip("/")


def airflow_api_bases():
    running_in_docker = Path("/.dockerenv").exists()
    candidates = [_AIRFLOW_API_HINT]

    if running_in_docker:
        candidates.extend([
            "http://airflow-apiserver:8080/api/v2",
            "http://localhost:8080/api/v2",
        ])
    else:
        candidates.extend([
            "http://localhost:8080/api/v2",
            "http://airflow-apiserver:8080/api/v2",
        ])

    return [base for i, base in enumerate(candidates) if base and base not in candidates[:i]]


def airflow_request(method, path, payload=None, timeout=10):
    auth = base64.b64encode(
        f"{AIRFLOW_API_USERNAME}:{AIRFLOW_API_PASSWORD}".encode("utf-8")
    ).decode("utf-8")
    errors = []

    for base in airflow_api_bases():
        url = f"{base}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url, data=data, method=method.upper())
        req.add_header("Authorization", f"Basic {auth}")
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")

        try:
            with urlrequest.urlopen(req, timeout=timeout) as response:
                raw = response.read()
                body = json.loads(raw.decode("utf-8")) if raw else {}
                return body, base
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            errors.append(f"{exc.code} from {url}: {body or exc.reason}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__} from {url}: {exc}")

    raise RuntimeError("Airflow API unavailable. Tried: " + " | ".join(errors))


def extract_collection(payload, *keys):
    for key in keys:
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            return payload[key]
    return []


def summarize_dag_run(run):
    if not isinstance(run, dict):
        return {}

    return {
        "dag_run_id": run.get("dag_run_id") or run.get("dagRunId") or run.get("run_id"),
        "state": run.get("state"),
        "run_type": run.get("run_type") or run.get("runType"),
        "logical_date": run.get("logical_date") or run.get("logicalDate"),
        "start_date": run.get("start_date") or run.get("startDate"),
        "end_date": run.get("end_date") or run.get("endDate"),
        "note": run.get("note"),
    }


def fetch_retraining_status(limit=5):
    payload, base = airflow_request(
        "GET",
        f"/dags/{AIRFLOW_DAG_ID}/dagRuns?limit={limit}&order_by=-start_date",
    )

    runs = [summarize_dag_run(run) for run in extract_collection(payload, "dag_runs", "dagRuns")]
    latest = runs[0] if runs else None
    tasks = []

    if latest and latest.get("dag_run_id"):
        try:
            task_payload, _ = airflow_request(
                "GET",
                f"/dags/{AIRFLOW_DAG_ID}/dagRuns/{latest['dag_run_id']}/taskInstances",
            )
            raw_tasks = extract_collection(task_payload, "task_instances", "taskInstances")
            tasks = [
                {
                    "task_id": task.get("task_id") or task.get("taskId"),
                    "state": task.get("state"),
                    "start_date": task.get("start_date") or task.get("startDate"),
                    "end_date": task.get("end_date") or task.get("endDate"),
                }
                for task in raw_tasks
            ]
        except Exception as exc:
            logger.warning(f"Could not fetch Airflow task instances: {exc}")

    return {
        "dag_id": AIRFLOW_DAG_ID,
        "api_available": True,
        "api_base": base,
        "latest_run": latest,
        "recent_runs": runs,
        "task_instances": tasks,
    }


def summarize_dvc_status_output(output):
    text = str(output or "").strip()
    if not text:
        text = "Pipeline is up to date."

    if "up to date" in text.lower():
        return {
            "status": "up_to_date",
            "summary": "Pipeline is up to date",
            "clean": True,
            "entries": [],
            "raw_output": text,
        }

    entries = []
    current_target = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if not line.startswith((" ", "\t")) and stripped.endswith(":"):
            current_target = stripped[:-1]
            continue

        if current_target and ":" in stripped:
            category, value = stripped.split(":", 1)
            entries.append(
                {
                    "target": current_target,
                    "category": category.strip(),
                    "value": value.strip(),
                }
            )
            continue

        entries.append(
            {
                "target": current_target or "pipeline",
                "category": "detail",
                "value": stripped,
            }
        )

    return {
        "status": "needs_update",
        "summary": f"{len(entries)} pending DVC change(s)",
        "clean": False,
        "entries": entries,
        "raw_output": text,
    }


def count_feedback_files(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def normalize_label_folder(name):
    normalized = str(name).strip().lower()
    if normalized == "parasitized":
        return "Parasitized"
    if normalized == "uninfected":
        return "Uninfected"
    return None


def make_unique_destination(base_dir, filename):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    candidate = base_dir / Path(filename).name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        alternate = base_dir / f"{stem}_{counter}{suffix}"
        if not alternate.exists():
            return alternate
        counter += 1


def extract_labelled_zip_to_feedback(zip_bytes):
    extracted_counts = {"Parasitized": 0, "Uninfected": 0}
    discovered_labels = set()
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]

        for member in members:
            parts = [
                part for part in Path(member.filename).parts
                if part and part != "__MACOSX" and not part.startswith("._")
            ]
            if not parts:
                continue

            label_name = next((normalize_label_folder(part) for part in parts if normalize_label_folder(part)), None)
            if label_name:
                discovered_labels.add(label_name)

        missing = sorted({"Parasitized", "Uninfected"} - discovered_labels)
        if missing:
            raise ValueError(
                "The labelled ZIP must contain both 'parasitized' and 'uninfected' subfolders."
            )

        for member in members:
            parts = [
                part for part in Path(member.filename).parts
                if part and part != "__MACOSX" and not part.startswith("._")
            ]
            if not parts:
                continue

            label_name = next((normalize_label_folder(part) for part in parts if normalize_label_folder(part)), None)
            if not label_name:
                continue

            filename = Path(parts[-1]).name
            if not filename:
                continue

            if Path(filename).suffix.lower() not in image_suffixes:
                continue

            with archive.open(member) as source:
                data = source.read()
                if not data:
                    continue

            destination = make_unique_destination(LABELLED_DIR / label_name, filename)
            with open(destination, "wb") as target:
                target.write(data)

            extracted_counts[label_name] += 1

    total = sum(extracted_counts.values())
    if total == 0:
        raise ValueError("No labelled image files were found inside the ZIP archive.")

    return {
        "total_extracted": total,
        "extracted_by_class": extracted_counts,
        "labelled_root": str(LABELLED_DIR),
    }


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

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.environ.get("MODEL_URI", "models:/MalariaClassifier/Production")
MODEL_REFRESH_INTERVAL_SECONDS = int(os.environ.get("MODEL_REFRESH_INTERVAL_SECONDS", "60"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger.info("Loading model from MLflow Production...")
logger.info(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")

client = MlflowClient()
model = None
latest_versions = []
run_id = None
version = None
run_name = "unknown"
f1 = None
acc = None
model_last_refresh_at = None
_last_model_refresh_check = 0.0
_model_lock = threading.RLock()


def fetch_latest_production_model():
    latest = client.get_latest_versions(
        name="MalariaClassifier",
        stages=["Production"]
    )
    if not latest:
        return None, []

    version_info = latest[0]
    active_run_id = version_info.run_id
    active_version = version_info.version
    run = client.get_run(active_run_id)

    return {
        "latest_versions": latest,
        "run_id": active_run_id,
        "version": active_version,
        "run_name": run.data.tags.get("mlflow.runName", "unknown"),
        "f1": run.data.metrics.get("test_f1"),
        "acc": run.data.metrics.get("test_accuracy"),
    }, latest


def refresh_model_if_needed(force=False):
    global model, latest_versions, run_id, version, run_name, f1, acc
    global model_last_refresh_at, _last_model_refresh_check

    now = time.time()
    if not force and (now - _last_model_refresh_check) < MODEL_REFRESH_INTERVAL_SECONDS:
        return False

    with _model_lock:
        if not force and (now - _last_model_refresh_check) < MODEL_REFRESH_INTERVAL_SECONDS:
            return False

        _last_model_refresh_check = now

        try:
            metadata, latest = fetch_latest_production_model()
            if metadata is None:
                logger.warning("No Production model found in MLflow.")
                return False

            current_version = str(version) if version is not None else None
            latest_version = str(metadata["version"])

            if force or model is None or current_version != latest_version:
                loaded_model = mlflow.pytorch.load_model(MODEL_URI)
                loaded_model.eval()

                model = loaded_model
                latest_versions = metadata["latest_versions"]
                run_id = metadata["run_id"]
                version = metadata["version"]
                run_name = metadata["run_name"]
                f1 = metadata["f1"]
                acc = metadata["acc"]
                model_last_refresh_at = datetime.now(timezone.utc).isoformat()

                if version is not None:
                    model_version.set(int(version))

                logger.info("🚀 Serving MLflow Production Model")
                logger.info(f"Version: {version}")
                logger.info(f"Run Name: {run_name}")
                logger.info(f"Run ID: {run_id}")
                logger.info(f"F1 Score: {f1}")
                logger.info(f"Accuracy: {acc}")
                return True

            model_last_refresh_at = datetime.now(timezone.utc).isoformat()
            latest_versions = latest
            return False

        except Exception as exc:
            logger.error(f"Model refresh failed: {exc}")
            return False


def get_model_sync_status(force_refresh=False):
    try:
        metadata, _ = fetch_latest_production_model()
    except Exception as exc:
        logger.warning(f"Could not inspect Production model status: {exc}")
        metadata = None

    latest_production_version = metadata["version"] if metadata else None
    latest_production_run_id = metadata["run_id"] if metadata else None
    serving_version = version
    stale = (
        latest_production_version is not None
        and str(latest_production_version) != str(serving_version)
    )

    refreshed = False
    if force_refresh and stale:
        refreshed = refresh_model_if_needed(force=True)
        serving_version = version
        stale = (
            latest_production_version is not None
            and str(latest_production_version) != str(serving_version)
        )

    return {
        "serving_version": serving_version,
        "latest_production_version": latest_production_version,
        "latest_production_run_id": latest_production_run_id,
        "model_loaded": model is not None,
        "stale": stale,
        "refreshed": refreshed,
        "last_refresh_at": model_last_refresh_at,
    }


def latest_run_promoted_successfully(status_payload):
    latest_run = (status_payload or {}).get("latest_run") or {}
    if latest_run.get("state") != "success":
        return False

    for task in (status_payload or {}).get("task_instances", []):
        if task.get("task_id") == "promote_model" and task.get("state") == "success":
            return True

    return False


refresh_model_if_needed(force=True)


_dvc_run_lock = threading.Lock()
_dvc_run_state = {
    "running": False,
    "current_stage": None,
    "started_at": None,
    "finished_at": None,
    "return_code": None,
    "status": "idle",
    "recent_logs": deque(maxlen=200),
    "last_command": "dvc repro",
}


def _append_dvc_log(line):
    cleaned = str(line).rstrip("\n")
    if cleaned:
        _dvc_run_state["recent_logs"].append(cleaned)


def _set_dvc_state(**updates):
    with _dvc_run_lock:
        _dvc_run_state.update(updates)


def _run_dvc_repro_background():
    _set_dvc_state(
        running=True,
        current_stage=None,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        return_code=None,
        status="running",
    )
    _dvc_run_state["recent_logs"].clear()
    _append_dvc_log("Starting `dvc repro` inside the backend container...")

    try:
        process = subprocess.Popen(
            ["dvc", "repro"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        stage_pattern = re.compile(r"Running stage '([^']+)'")

        if process.stdout is not None:
            for line in process.stdout:
                _append_dvc_log(line)
                match = stage_pattern.search(line)
                if match:
                    _set_dvc_state(current_stage=match.group(1))

        return_code = process.wait()
        status = "completed" if return_code == 0 else "failed"
        _set_dvc_state(
            running=False,
            finished_at=datetime.now(timezone.utc).isoformat(),
            return_code=return_code,
            status=status,
        )

        if return_code == 0:
            _append_dvc_log("DVC repro completed successfully.")
            retraining_triggered_total.inc()
            refresh_model_if_needed(force=True)
        else:
            _append_dvc_log(f"DVC repro failed with exit code {return_code}.")

    except Exception as exc:
        _append_dvc_log(f"Failed to start DVC repro: {exc}")
        _set_dvc_state(
            running=False,
            finished_at=datetime.now(timezone.utc).isoformat(),
            return_code=-1,
            status="failed",
        )


def start_dvc_repro():
    with _dvc_run_lock:
        if _dvc_run_state["running"]:
            return False

        thread = threading.Thread(target=_run_dvc_repro_background, daemon=True)
        thread.start()
        return True


def get_dvc_repro_status():
    with _dvc_run_lock:
        return {
            "running": _dvc_run_state["running"],
            "status": _dvc_run_state["status"],
            "current_stage": _dvc_run_state["current_stage"],
            "started_at": _dvc_run_state["started_at"],
            "finished_at": _dvc_run_state["finished_at"],
            "return_code": _dvc_run_state["return_code"],
            "recent_logs": list(_dvc_run_state["recent_logs"]),
            "last_command": _dvc_run_state["last_command"],
        }


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
    refresh_model_if_needed()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": version,
        "model_last_refresh_at": model_last_refresh_at,
    }


@app.get("/ready")
def ready():
    refresh_model_if_needed()
    return {
        "ready": model is not None,
        "model_loaded": model is not None,
        "model_version": version,
    }


@app.middleware("http")
async def track_requests(request, call_next):
    start = time.time()
    refresh_model_if_needed()

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
        refresh_model_if_needed()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Production model is not loaded yet"
            )

        logger.info(f"Received file: {file.filename}")

        # -------- INPUT VALIDATION -------- #
        '''if file.content_type not in ["image/jpeg", "image/png"]:
            log_failure({
                "filename": file.filename,
                "type": "invalid_input",
                "reason": "unsupported_format",
                "timestamp": time.time()
            })
            invalid_image_uploads_total.inc()
            raise HTTPException(status_code=422, detail="Only JPG and PNG images are supported")'''

        image_bytes = await file.read()

        if not image_bytes:
            log_failure({
                "filename": file.filename,
                "type": "invalid_input",
                "reason": "empty_file",
                "timestamp": time.time()
            })
            invalid_image_uploads_total.inc()
            _global_counts["invalid"] += 1
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            log_failure({
                "filename": file.filename,
                "type": "invalid_input",
                "reason": "not_an_image_or_corrupt",
                "timestamp": time.time()
            })

            invalid_image_uploads_total.inc()

            _global_counts["invalid"] += 1

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

        with _model_lock:
            with torch.no_grad():
                outputs = model(input_tensor)

                probs = torch.softmax(outputs, dim=1)
                pred = probs.argmax(dim=1).item()
                confidence = probs.max().item()
                confidence_sum.inc(confidence)
                confidence_count.inc()
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
            _global_counts["ood"] += 1
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
        cpu_usage.set(psutil.cpu_percent())

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
                confidence,
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
            "ood_count": _global_counts["ood"],
            "invalid_count": _global_counts["invalid"],
            "confidence_samples": _confidence_stats["count"],
        }


    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to compute metrics"
        )


@app.get("/api/feedback/stats")
def feedback_stats():
    try:
        labelled_by_class = {
            class_name: count_feedback_files(LABELLED_DIR / class_name)
            for class_name in params["data"]["classes"]
        }

        return {
            "unlabeled_count": count_feedback_files(UNLABELED_DIR),
            "unlabeled_ood_count": count_feedback_files(UNLABELED_OOD_DIR),
            "labelled_total": sum(labelled_by_class.values()),
            "labelled_by_class": labelled_by_class,
            "labelled_root": str(LABELLED_DIR),
            "unlabeled_root": UNLABELED_DIR,
            "unlabeled_ood_root": UNLABELED_OOD_DIR,
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to compute feedback stats"
        )


@app.post("/api/feedback/upload-labelled-zip")
async def upload_labelled_zip(file: UploadFile = File(...)):
    try:
        if not file.filename or not file.filename.lower().endswith(".zip"):
            raise HTTPException(
                status_code=400,
                detail="Please upload a .zip file containing labelled feedback."
            )

        zip_bytes = await file.read()
        result = extract_labelled_zip_to_feedback(zip_bytes)

        uploaded_total = result["total_extracted"]
        if uploaded_total:
            labelled_images_received_total.inc(uploaded_total)

        return {
            "status": "success",
            "message": "Labelled feedback uploaded successfully.",
            **result,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid ZIP archive.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to ingest labelled ZIP: {exc}")
        raise HTTPException(status_code=500, detail="Failed to ingest labelled ZIP.")


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
    if start_dvc_repro():
        return {
            "status": "started",
            "message": "DVC repro started in the background.",
            "repro_status": get_dvc_repro_status(),
        }

    return {
        "status": "already_running",
        "message": "A DVC repro run is already in progress.",
        "repro_status": get_dvc_repro_status(),
    }


@app.post("/api/retraining/trigger")
def trigger_retraining_dag():
    try:
        payload, _ = airflow_request(
            "POST",
            f"/dags/{AIRFLOW_DAG_ID}/dagRuns",
            payload={
                "conf": {"trigger_source": "frontend"},
            },
        )
        retraining_triggered_total.inc()
        return {
            "status": "triggered",
            "dag_id": AIRFLOW_DAG_ID,
            "dag_run_id": payload.get("dag_run_id") or payload.get("dagRunId"),
            "state": payload.get("state"),
            "message": "Airflow retraining DAG triggered successfully",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger Airflow DAG: {exc}"
        )


@app.get("/api/retraining/status")
def retraining_status():
    try:
        status_payload = fetch_retraining_status()
        status_payload["model_sync"] = get_model_sync_status(
            force_refresh=latest_run_promoted_successfully(status_payload)
        )
        return status_payload
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch Airflow retraining status: {exc}"
        )


@app.post("/api/model/refresh")
def refresh_serving_model():
    try:
        return get_model_sync_status(force_refresh=True)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh serving model: {exc}"
        )


# ---------------- PIPELINE STATUS ---------------- #

@app.get("/pipeline/status")
def pipeline_status():
    try:
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True,
            check=False,
        )

        output = (result.stdout + "\n" + result.stderr).strip()
        dvc_status = summarize_dvc_status_output(output)

        return {
            "dvc": dvc_status,
            "return_code": result.returncode,
            "repro_command": "docker compose exec backend dvc repro",
            "repro_status": get_dvc_repro_status(),
            "model_sync": get_model_sync_status(),
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
        refresh_model_if_needed()
        if latest_versions:
            if version is not None:
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

@app.post("/log_invalid")
def log_invalid(data: dict):
    log_failure({
        "filename": data["filename"],
        "type": "invalid_input",
        "reason": data["reason"],
        "timestamp": time.time()
    })
    invalid_image_uploads_total.inc()
    return {"status": "logged"}
