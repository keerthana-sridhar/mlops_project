import io
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
            raise HTTPException(status_code=500, detail="Model inference error")

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

        return {
            "dvc_status": result.stdout.strip(),
            "clean": result.returncode == 0
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
        )