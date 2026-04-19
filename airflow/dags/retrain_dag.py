from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta

PROJECT_PATH = "/opt/project"

with DAG(
    dag_id="malaria_retraining",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=2),
) as dag:

    run_dvc = BashOperator(
        task_id="run_dvc_pipeline",
        bash_command=f"cd {PROJECT_PATH} && dvc repro",
        execution_timeout=None,    # no task-level timeout
    )