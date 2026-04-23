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
    return len(os.listdir(FEEDBACK_PATH)) > 0


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

    snapshot = BashOperator(
        task_id="dvc_snapshot",
        bash_command="""
            cd /opt/project &&
            test -f finetune/checkpoint.pth &&
            dvc add finetune/checkpoint.pth &&
            dvc add data/processed/incremental_resized &&
            (dvc push || echo "DVC push skipped")
        """,
    )

 

    check >> process >> finetune >> evaluate >> promote >>  snapshot
