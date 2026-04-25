import os
import shutil
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = PROJECT_ROOT / "shared" / "mlflow"
ARTIFACT_DESTINATION = SHARED_ROOT / "artifacts"
DB_PATH = SHARED_ROOT / "mlflow.db"
MIGRATION_SENTINEL = SHARED_ROOT / ".migration_complete"

CANONICAL_ARTIFACT_URI_ROOT = os.environ.get(
    "MLFLOW_CANONICAL_ARTIFACT_ROOT",
    "mlflow-artifacts:/",
).rstrip("/")

LEGACY_DB_CANDIDATES = [
    PROJECT_ROOT / "mlflow.db",
    PROJECT_ROOT / "mlflow_docker.db",
]

LEGACY_ARTIFACT_CANDIDATES = [
    PROJECT_ROOT / "mlruns",
]


def ensure_dirs():
    ARTIFACT_DESTINATION.mkdir(parents=True, exist_ok=True)
    SHARED_ROOT.mkdir(parents=True, exist_ok=True)


def seed_database():
    if DB_PATH.exists():
        return

    for candidate in LEGACY_DB_CANDIDATES:
        if candidate.exists():
            shutil.copy2(candidate, DB_PATH)
            print(f"Seeded shared MLflow DB from {candidate}")
            return

    print("No legacy MLflow database found; MLflow server will initialize a new one.")


def merge_artifacts():
    for candidate in LEGACY_ARTIFACT_CANDIDATES:
        if candidate.exists():
            for child in candidate.iterdir():
                destination = ARTIFACT_DESTINATION / child.name
                if child.is_dir():
                    shutil.copytree(child, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(child, destination)
            print(f"Merged legacy MLflow artifacts from {candidate}")


def rewrite_artifact_uri(value):
    if not value:
        return value

    normalized = str(value).strip().replace("\\", "/")
    if not normalized:
        return normalized

    if normalized.startswith("file://"):
        normalized = normalized[len("file://") :]

    if normalized.startswith(f"{CANONICAL_ARTIFACT_URI_ROOT}/"):
        return normalized

    if "/mlruns/" in normalized:
        suffix = normalized.split("/mlruns/", 1)[1].lstrip("/")
        return f"{CANONICAL_ARTIFACT_URI_ROOT}/{suffix}"

    if normalized.startswith("mlruns/"):
        suffix = normalized[len("mlruns/") :].lstrip("/")
        return f"{CANONICAL_ARTIFACT_URI_ROOT}/{suffix}"

    return normalized


def rewrite_database_paths():
    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    updates = 0

    for experiment_id, artifact_location in cur.execute(
        "SELECT experiment_id, artifact_location FROM experiments"
    ).fetchall():
        updated = rewrite_artifact_uri(artifact_location)
        if updated != artifact_location:
            cur.execute(
                "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                (updated, experiment_id),
            )
            updates += 1

    for run_uuid, artifact_uri in cur.execute(
        "SELECT run_uuid, artifact_uri FROM runs"
    ).fetchall():
        updated = rewrite_artifact_uri(artifact_uri)
        if updated != artifact_uri:
            cur.execute(
                "UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?",
                (updated, run_uuid),
            )
            updates += 1

    for name, version, storage_location in cur.execute(
        "SELECT name, version, storage_location FROM model_versions"
    ).fetchall():
        updated = rewrite_artifact_uri(storage_location)
        if updated != storage_location:
            cur.execute(
                "UPDATE model_versions SET storage_location = ? WHERE name = ? AND version = ?",
                (updated, name, version),
            )
            updates += 1

    conn.commit()
    conn.close()
    print(f"Rewrote {updates} MLflow artifact path entries")


def main():
    ensure_dirs()
    if MIGRATION_SENTINEL.exists() and DB_PATH.exists():
        print("Shared MLflow store already initialized; skipping legacy migration.")
        return

    seed_database()
    merge_artifacts()
    rewrite_database_paths()
    MIGRATION_SENTINEL.write_text("ok\n")


if __name__ == "__main__":
    main()
