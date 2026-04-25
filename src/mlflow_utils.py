import os
import subprocess
from pathlib import Path

import mlflow


DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def configure_mlflow(tracking_uri=None):
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)


def _safe_git_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def get_git_commit():
    return _safe_git_output(["git", "rev-parse", "HEAD"])


def get_git_branch():
    return _safe_git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def log_reproducibility_tags(extra_tags=None):
    tags = {
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
        "runtime_origin": "docker" if Path("/.dockerenv").exists() else "host",
    }
    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags(tags)
