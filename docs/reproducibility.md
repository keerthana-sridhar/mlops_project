# Reproducibility and Portability

## Current state

The codebase is now much closer to portable-by-default:

- MLflow runs are stored in a shared tracking database at `shared/mlflow/mlflow.db`
- artifacts are copied into `shared/mlflow/artifacts`
- the `mlflow` container serves the UI on port `5000` against that same shared store

That means new runs triggered from the backend, Airflow, or the reproducible Docker stack all land in one shared place instead of mixing host-local and container-local paths.

Git push alone is still **not enough** to reproduce everything on another laptop, because DVC data and local secrets still need machine-specific setup.

## What to push to Git

Push the following:

- source code under `src/`, `frontend/`, `airflow/`, `docker/`, and `monitoring/`
- `docker-compose.yaml`
- `requirements.txt`
- `params.yaml`
- `dvc.yaml`
- `dvc.lock`
- `.dvc/` metadata, except `.dvc/config.local`, `.dvc/tmp`, and `.dvc/cache`
- documentation under `docs/`

Do **not** push:

- `.env`
- `.dvc/config.local`
- `mlflow.db`
- `mlruns/`
- `shared/mlflow/`
- `logs/`
- generated reports or transient runtime files unless you intentionally want them versioned

## What is needed on another laptop

1. Clone the Git repository.
2. Create `.env` from `.env.example`.
3. Configure a **shared DVC remote** on both laptops.
4. Pull DVC-tracked data and artifacts.
5. Start the stack with Docker Compose.
6. Open MLflow at `http://localhost:5000` on the same laptop, or replace `localhost` with the host machine IP from another laptop on the same network.

## Recommended DVC setup

Use a shared remote such as:

- S3
- Google Drive
- SSH/SFTP
- an institutional file server

Keep the remote configuration out of Git by storing it in `.dvc/config.local`.

Example pattern:

```bash
dvc remote add -d storage <shared-remote-url>
dvc remote modify --local storage <key> <value>
dvc pull
```

This keeps secrets off the repository while still allowing another laptop to reproduce the tracked pipeline state.

## MLflow portability

The stack now uses a dedicated MLflow server service for the UI, while Docker services log directly into the shared backend store.

How it works:

- `docker/mlflow-entrypoint.sh` starts the tracking server UI
- `src/migrate_mlflow_store.py` copies legacy `mlflow.db` and `mlruns/` data into `shared/mlflow/`
- the script rewrites artifact URIs to `mlflow-artifacts:/...` so the server can serve them cleanly from any laptop that can reach the host

For local terminal runs outside Docker, point your shell to the server first:

```bash
export MLFLOW_TRACKING_URI=sqlite:////opt/project/shared/mlflow/mlflow.db
```

Inside Docker Compose, the services already use the shared SQLite tracking URI.

## Do secrets need to be exposed in the terminal or UI?

No.

Use:

- `.env` for container environment variables
- `.dvc/config.local` for machine-specific DVC credentials

Those files stay local and should not be committed.
