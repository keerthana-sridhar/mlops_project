from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import ShortCircuitOperator
from datetime import datetime, timedelta
import os

PROJECT_PATH = "/opt/project"
FEEDBACK_PATH = f"{PROJECT_PATH}/data/feedback/labelled"
PYTHON_CMD = "PYTHONUNBUFFERED=1 python"

def check_new_data():
    if not os.path.exists(FEEDBACK_PATH):
        return False
    
    # Check subdirectories for actual image files, not just folder existence
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for root, dirs, files in os.walk(FEEDBACK_PATH):
        for f in files:
            if os.path.splitext(f)[1].lower() in image_extensions:
                return True
    return False


with DAG(
    dag_id="malaria_retraining_v2",
    start_date=datetime(2026, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=2),
) as dag:

    check = ShortCircuitOperator(
        task_id="check_new_images",
        python_callable=check_new_data,
    )

    process = BashOperator(
        task_id="process_feedback",
        bash_command=f"cd {PROJECT_PATH} && {PYTHON_CMD} src/process_feedback.py",
    )

    finetune = BashOperator(
        task_id="finetune_model",
        bash_command=f"cd {PROJECT_PATH} && {PYTHON_CMD} src/finetune.py",
    )

    evaluate = BashOperator(
        task_id="evaluate_finetune",
        bash_command=f"cd {PROJECT_PATH} && {PYTHON_CMD} src/eval_finetune.py",
    )

    promote = BashOperator(
        task_id="promote_model",
        bash_command=f"cd {PROJECT_PATH} && {PYTHON_CMD} src/promote_model.py",
    )

    check >> process >> finetune >> evaluate >> promote
