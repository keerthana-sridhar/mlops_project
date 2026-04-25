import streamlit as st
import requests
import pandas as pd
from PIL import Image
import time
import os
import zipfile
import io
import json
import matplotlib.pyplot as plt
import hashlib

import mimetypes

# ── CONFIG ─────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Malaria Diagnostic System",
    page_icon="🧬",
    layout="wide"
)

# ── SIDEBAR ────────────────────────────
st.sidebar.title("🧬 Malaria Diagnostic System")
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Live Metrics")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Pipeline", "Experiments", "User Guide"]
)
# ---------------- LIVE METRICS IN SIDEBAR ---------------- #
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Live Metrics")

try:
    res = requests.get(f"{BACKEND_URL}/api/metrics")
    metrics = res.json()

    st.sidebar.metric(
        "Total Predictions",
        metrics.get("total_predictions", 0)
    )
    st.sidebar.metric(
    "OOD Count",
    metrics.get("ood_count", 0)
)

    st.sidebar.metric(
        "Invalid Inputs",
        metrics.get("invalid_count", 0)
    )

    st.sidebar.metric(
        "Avg Confidence",
        round(metrics.get("average_confidence", 0), 3)
    )

    st.sidebar.metric(
        "Low Conf Rate",
        round(metrics.get("low_confidence_rate", 0), 3)
    )

    if metrics.get("drift_detected"):
        st.sidebar.error("⚠️ Drift")
    else:
        st.sidebar.success("Stable")

except Exception:
    st.sidebar.warning("Metrics unavailable")

# ---------------- FAILURE LOG IN SIDEBAR ---------------- #
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Failure Log")

with st.sidebar.expander("📁 Failure Log", expanded=False):
    try:
        with open("failure.json", "r") as f:
            failure_data = json.load(f)

        if failure_data:
            # show recent entries
            for item in failure_data[-10:]:
                label = item.get("type", "unknown")
                filename = item.get("filename", "unknown")

                if label == "invalid_input":
                    reason = item.get("reason", "")
                    st.write(f"🔴 {filename} (invalid: {reason})")

                elif label == "ood":
                    reason = item.get("reason", "")
                    st.write(f"🟠 {filename} (ood: {reason})")

                elif label == "uncertain":
                    entropy = round(item.get("entropy", 0), 3)
                    st.write(f"🟡 {filename} (uncertain, entropy={entropy})")

                else:
                    st.write(f"• {filename} ({label})")
            # -------- DOWNLOAD BUTTON -------- #
            st.download_button(
                label="⬇️ Download Failure Log",
                data=json.dumps(failure_data, indent=2),
                file_name="failure_log.json",
                mime="application/json"
            )
                    # -------- CLEAR FAILURE LOG -------- #
            if st.button("🗑️ Clear Failure Log"):
                try:
                    with open("failure.json", "w") as f:
                        json.dump([], f)

                    st.success("Failure log cleared")
                    st.rerun()

                except Exception:
                    st.error("Failed to clear log")

        else:
            st.caption("No failures yet")

    except Exception:
        st.caption("Failure log unavailable")



# ── USER GUIDE ─────────────────────────
if page == "User Guide":
    st.title("📘 User Guide")

    st.markdown("""
    ### What This App Does

    This system predicts whether a blood smear cell image is **Parasitized** or **Uninfected**, monitors live model quality, and supports a feedback loop for retraining.

    ### How to Use

    1. Open **Home**
    2. Upload JPG, PNG, or ZIP files containing images
    3. Click **Run Analysis**
    4. Review prediction labels, confidence scores, latency, and alerts

    ### Understanding OOD and Uncertain Cases

    - If the model detects an **OOD** (out-of-distribution) or **uncertain** sample, the file is saved into the feedback queue.
    - If many OOD or uncertain files accumulate, they will appear in the unlabelled queues and should be manually reviewed before retraining.
    - These files are stored under:
      - `data/feedback/unlabeled`
      - `data/feedback/unlabeled_ood`
    - A physician or domain expert should manually annotate those images and move them into:
      - `data/feedback/labelled/Parasitized`
      - `data/feedback/labelled/Uninfected`
    - You can also upload a `labelled.zip` from the **Pipeline** page. The ZIP must contain both `parasitized` and `uninfected` subfolders.

    ### Retraining Workflow

    - Use the **Pipeline** page to trigger retraining through **Airflow**.
    - The UI shows the latest Airflow DAG run, task-by-task progress, and the current **DVC reproducibility status**.
    - When a newer MLflow Production model is available, the backend refreshes the serving model automatically.

    ### Running The Full Reproducible DVC Pipeline In Docker

    Run this from the project root:

    ```bash
    docker compose exec backend dvc repro
    ```

    This is reproducible because the command runs inside the same containerized environment, using the same tracked code, parameters, and mounted project workspace.
    """)

# ── HOME ───────────────────────────────
elif page == "Home":

    st.title("🧬 Malaria Cell Classification")
    st.markdown("### AI-assisted diagnostic tool")

    # Backend check
    try:
        requests.get(f"{BACKEND_URL}/health")
        st.success("Backend connected ✅")
    except:
        st.error("Backend is not running ❌")
        st.stop()
        # -------- SYSTEM ALERTS (FROM ALERTMANAGER) -------- #
    st.subheader("🚨 System Alerts")

    try:
        alert_res = requests.get(f"{BACKEND_URL}/alerts")
        alert_data = alert_res.json()
        alerts = alert_data.get("alerts", [])

        if not alerts:
            st.success("✅ No active system alerts")
        else:
            for alert in alerts:
                name = alert["labels"].get("alertname", "Unknown")
                severity = alert["labels"].get("severity", "warning")

                summary = alert.get("annotations", {}).get("summary", "")
                description = alert.get("annotations", {}).get("description", "")

                if severity == "critical":
                    st.error(f"🚨 {summary}\n{description}")
                else:
                    st.warning(f"⚠️ {summary}\n{description}")

    except Exception:
        st.warning("Could not fetch system alerts")
    

    uploaded_files = st.file_uploader(
        "Upload Images or ZIP",
        type=["png", "jpg", "jpeg", "zip"],
        accept_multiple_files=True
    )

    MAX_SIZE_MB = 10

    valid_files = []
    failed_files = []
    duplicate_files = []
    seen_upload_hashes = set()

    # -------- PROCESS INPUTS -------- #

    def report_invalid_input(filename, reason):
        failed_files.append({"filename": filename, "reason": reason})
        try:
            requests.post(
                f"{BACKEND_URL}/log_invalid",
                json={"filename": filename, "reason": reason},
                timeout=5,
            )
        except Exception:
            pass
    
    def process_file(file, name_override=None):
        filename = name_override or file.name

        try:
            file_bytes = file.getvalue()
            if not file_bytes:
                report_invalid_input(filename, "empty_file")
                return

            if len(file_bytes) > MAX_SIZE_MB * 1024 * 1024:
                report_invalid_input(filename, "file_too_large")
                return

            file_hash = hashlib.sha256(file_bytes).hexdigest()
            if file_hash in seen_upload_hashes:
                duplicate_files.append(filename)
                return
            seen_upload_hashes.add(file_hash)

            try:
                image = Image.open(io.BytesIO(file_bytes))
                image.verify()

                # reopen fresh buffer
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

            except Exception:
                image = None

        except Exception:
            report_invalid_input(filename, "read_failed")
            return

        valid_files.append((filename, file_bytes, image, image is not None))

    if uploaded_files:

        # Handle ZIP + normal files
        for file in uploaded_files:

            if file.name.endswith(".zip"):
                try:
                    zip_bytes = io.BytesIO(file.getvalue())
                    with zipfile.ZipFile(zip_bytes) as z:
                        for name in z.namelist():
                            if name.endswith("/"):
                                continue
                            if "__MACOSX" in name or os.path.basename(name).startswith("._"):
                                continue

                            extracted = io.BytesIO(z.read(name))
                            extracted.name = name
                            process_file(extracted, name_override=name)
                except Exception:
                    requests.post(
                        f"{BACKEND_URL}/log_invalid",
                        json={
                            "filename": file.name,
                            "reason": "invalid_zip"
                        }
                    )
            else:
                process_file(file)

        # -------- PREVIEW -------- #
        st.subheader("Preview")

        if valid_files:
            cols = st.columns(min(4, len(valid_files)))

            for i, (_, _, image, is_valid) in enumerate(valid_files):
                if image is not None:
                    cols[i % 4].image(image, use_container_width=True)
                else:
                    cols[i % 4].warning("Invalid image ❌")
        else:
            st.warning("No valid images to preview (all files are corrupt or invalid)")

        # Show failed files early
        if failed_files:
            st.warning(f"{len(failed_files)} files skipped ❌")
        if duplicate_files:
            st.info(f"{len(duplicate_files)} duplicate uploads were ignored to prevent double-counting.")

        # -------- RUN ANALYSIS -------- #
        if st.button("🔍 Run Analysis"):

            if not valid_files:
                st.error("No valid images to process")
                st.stop()

            results = []

            with st.spinner(f"Processing {len(valid_files)} images..."):

                for filename, file_bytes, _, is_valid in valid_files:
                    try:
                        files = {
                            "file": (
                                filename,
                                file_bytes,
                                mimetypes.guess_type(filename)[0] or "application/octet-stream",
                            )
                        }
                        start = time.time()
                        response = requests.post(
                            f"{BACKEND_URL}/predict",
                            files=files
                        )
                        latency = time.time() - start

                        if response.status_code == 200:
                            res = response.json()

                            label = res.get("prediction_label")

                            if label == "ood":
                                status = "warning"
                                alert = "⚠️ OOD detected — check failure log & consider retraining"

                            elif label == "uncertain":
                                status = "warning"
                                alert = "⚠️ Model uncertain — review sample in failure log"

                            elif label == "invalid_input":
                                status = "error"
                                alert = "❌ Invalid file — see reason in failure log"

                            else:
                                status = "success"
                                alert = "✅ Prediction OK"
                            label = res.get("prediction_label")
                            confidence = res.get("confidence")
                            if confidence is not None:
                                confidence = round(confidence, 3)

                            results.append({
    "filename": filename,
    "prediction": label,
    "confidence": confidence,
    "latency": round(latency, 3),
    "status": status,
    "alert": alert,
    "message": res.get("message", "")
})

                        else:
                            results.append({
                                "filename": filename,
                                "status": "error",
                                "alert": "❌ Invalid input (caught by backend)"
                            })

                    except Exception:
                        results.append({
                            "filename": filename,
                            "status": "failed"
                        })

            st.success("Analysis Complete")

            # -------- RESULTS -------- #
            df = pd.DataFrame(results)
            st.subheader("📊 Results Table")
            st.dataframe(df)
            # -------- ALERT DISPLAY -------- #
            st.subheader("⚠️ Alerts")

            for r in results:
                if r["status"] == "warning":
                    st.warning(f"{r['filename']}: {r['alert']} — {r['message']}")
                elif r["status"] == "error":
                    msg = r.get("message", "No additional details")
                    st.error(f"{r['filename']}: {r['alert']} — {msg}")
            st.subheader("📊 Live Model Monitoring")

            try:
                res = requests.get(f"{BACKEND_URL}/api/metrics")
                metrics = res.json()

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Predictions", metrics.get("total_predictions", 0))
                col2.metric("Avg Confidence", round(metrics.get("average_confidence", 0), 3))
                col3.metric("Low Confidence Rate", round(metrics.get("low_confidence_rate", 0), 3))

                if metrics.get("drift_detected"):
                    st.error("⚠️ Drift Detected — Model performance degrading")
                else:
                    st.success("✅ Model Stable")

            except Exception:
                st.warning("Metrics unavailable")

            # -------- PLOT -------- #
            success_count = df[df["status"] == "success"].shape[0]
            fail_count = len(failed_files)

            st.subheader("📈 Success vs Failed")

            fig, ax = plt.subplots()
            ax.bar(["Success", "Failed"], [success_count, fail_count])
            ax.set_ylabel("Count")
            st.pyplot(fig)

            

            
    

# ── PIPELINE ───────────────────────────
elif page == "Pipeline":
    st.title("⚙️ ML Pipeline")

    st.subheader("📌 Pipeline Architecture")
    st.image("frontend/assets/pipeline.png")

    st.caption("Airflow orchestrates feedback retraining, while DVC reports the reproducible CI pipeline state for tracked code, data, and reports.")

    st.subheader("🔁 DVC Pipeline DAG")
    st.image("frontend/assets/dvc_dag.png")

    st.subheader("📦 Feedback Queue")
    try:
        feedback_res = requests.get(f"{BACKEND_URL}/api/feedback/stats", timeout=10)
        feedback_res.raise_for_status()
        feedback = feedback_res.json()

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Unlabelled Queue", feedback.get("unlabeled_count", 0))
        q2.metric("OOD Queue", feedback.get("unlabeled_ood_count", 0))
        q3.metric("Labelled Total", feedback.get("labelled_total", 0))
        q4.metric("Parasitized Labels", feedback.get("labelled_by_class", {}).get("Parasitized", 0))

        st.caption(
            "Manual review workflow: clinicians should inspect files in the unlabelled queues, annotate them, and move them to "
            "`data/feedback/labelled/Parasitized` or `data/feedback/labelled/Uninfected`. "
            "If many OOD files accumulate, package the reviewed data as `labelled.zip` with both class folders and upload it below."
        )
    except Exception:
        st.warning("Feedback queue stats unavailable")

    st.subheader("📥 Upload Labelled Feedback ZIP")
    labelled_zip = st.file_uploader(
        "Upload `labelled.zip` containing `parasitized` and `uninfected` subfolders",
        type=["zip"],
        key="labelled_zip_upload",
    )

    if st.button("Upload labelled.zip", use_container_width=True):
        if labelled_zip is None:
            st.error("Choose a labelled ZIP file first.")
        else:
            with st.spinner("Validating and extracting labelled feedback..."):
                try:
                    upload_res = requests.post(
                        f"{BACKEND_URL}/api/feedback/upload-labelled-zip",
                        files={"file": (labelled_zip.name, labelled_zip.getvalue(), "application/zip")},
                        timeout=120,
                    )
                    data = upload_res.json()
                    if upload_res.status_code == 200:
                        st.success(data.get("message", "Labelled ZIP uploaded successfully."))
                        st.write("Extracted by class:", data.get("extracted_by_class", {}))
                    else:
                        st.error(data.get("detail", "Failed to upload labelled ZIP."))
                except Exception as exc:
                    st.error(f"Failed to upload labelled ZIP: {exc}")

    st.subheader("🚀 Airflow Retraining")
    control_col1, control_col2 = st.columns(2)

    if control_col1.button("Trigger Retraining DAG", use_container_width=True):
        with st.spinner("Triggering the Airflow retraining DAG..."):
            try:
                retrain_res = requests.post(f"{BACKEND_URL}/api/retraining/trigger", timeout=30)
                data = retrain_res.json()
                if retrain_res.status_code == 200:
                    st.success(data.get("message", "Airflow retraining DAG triggered"))
                    st.caption(f"Run ID: {data.get('dag_run_id')}")
                else:
                    st.error(data.get("detail", "Failed to trigger the retraining DAG"))
            except Exception as exc:
                st.error(f"Failed to trigger Airflow retraining: {exc}")

    control_col2.button("Refresh Status", use_container_width=True)

    st.subheader("📟 Orchestration Status")
    try:
        retraining_res = requests.get(f"{BACKEND_URL}/api/retraining/status", timeout=10)
        retraining_res.raise_for_status()
        retraining = retraining_res.json()

        model_sync = retraining.get("model_sync", {})
        if model_sync.get("stale"):
            refresh_res = requests.post(f"{BACKEND_URL}/api/model/refresh", timeout=20)
            if refresh_res.status_code == 200:
                model_sync = refresh_res.json()

        latest_run = retraining.get("latest_run") or {}
        task_instances = retraining.get("task_instances") or []

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Latest DAG State", str(latest_run.get("state", "idle")).title())
        s2.metric("Serving Version", model_sync.get("serving_version") or "-")
        s3.metric("Latest Production", model_sync.get("latest_production_version") or "-")
        s4.metric("Tracked Tasks", len(task_instances))

        if not latest_run:
            st.info("No Airflow retraining run has been recorded yet.")
        else:
            st.caption(
                f"Run ID: {latest_run.get('dag_run_id')} | Started: {latest_run.get('start_date') or '-'} | "
                f"Finished: {latest_run.get('end_date') or 'still running'}"
            )

            latest_state = str(latest_run.get("state", "")).lower()
            if latest_state == "success":
                st.success("Latest Airflow retraining run completed successfully.")
            elif latest_state in {"failed", "error"}:
                st.error("Latest Airflow retraining run failed.")
                if retraining.get("failed_task_id") or retraining.get("failure_reason"):
                    st.caption(
                        f"Failed task: {retraining.get('failed_task_id') or '-'} | "
                        f"Reason: {retraining.get('failure_reason') or 'No reason available'}"
                    )
            elif latest_state:
                st.info(f"Latest Airflow retraining run is `{latest_state}`.")

        if model_sync.get("refreshed"):
            st.success("Backend serving model refreshed to the latest Production version.")
        elif model_sync.get("stale"):
            st.warning("A newer Production model exists, but the backend is still serving the previous version.")

        if task_instances:
            task_df = pd.DataFrame(task_instances)
            st.subheader("🧩 DAG Task States")
            st.dataframe(task_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Could not fetch Airflow retraining status: {exc}")

    st.subheader("🗂️ DVC Reproducibility Status")
    try:
        res = requests.get(f"{BACKEND_URL}/pipeline/status", timeout=30)
        res.raise_for_status()
        data = res.json()
        dvc = data.get("dvc", {})
        repro = data.get("repro_status", {})
        dvc_state = dvc.get("status")
        if dvc_state == "up_to_date":
            dvc_label = "Up to date"
        elif dvc_state == "busy":
            dvc_label = "Busy"
        elif dvc_state == "unavailable":
            dvc_label = "Unavailable"
        else:
            dvc_label = "Needs attention"
        pending_value = "-" if dvc_state in {"busy", "unavailable"} else len(dvc.get("entries", []))

        d1, d2, d3 = st.columns(3)
        d1.metric("DVC Status", dvc_label)
        d2.metric("Pending Changes", pending_value)
        d3.metric("Legacy DVC Repro", str(repro.get("status", "idle")).title())

        if dvc_state == "busy":
            st.info(dvc.get("summary", "DVC is busy right now"))
        elif dvc_state == "unavailable":
            st.warning(dvc.get("summary", "DVC status is temporarily unavailable"))
        elif dvc.get("clean"):
            st.success(dvc.get("summary", "Pipeline is up to date"))
        else:
            st.warning(dvc.get("summary", "DVC changes are pending"))

        if dvc.get("source") == "cache":
            st.info(
                f"Showing the last successful DVC status from {dvc.get('checked_at') or 'an earlier check'} "
                f"because the live check failed: {dvc.get('live_error') or 'unknown reason'}"
            )
        elif dvc.get("checked_at"):
            st.caption(f"Checked at: {dvc.get('checked_at')}")

        if dvc.get("entries"):
            st.dataframe(pd.DataFrame(dvc["entries"]), use_container_width=True)

        with st.expander("Raw DVC Status"):
            st.code(dvc.get("raw_output", "No output"), language="bash")

    except Exception as exc:
        st.error(f"Could not fetch DVC status: {exc}")

# ── EXPERIMENTS ────────────────────────
elif page == "Experiments":
    st.title("📊 Model Information")

    try:
        res = requests.get(f"{BACKEND_URL}/model/info")
        data = res.json()

        if "message" in data:
            st.warning(data["message"])
        else:
            st.metric("Model", data.get("model_name"))
            st.metric("Accuracy", data.get("accuracy"))
            st.metric("F1 Score", data.get("f1_score"))

            st.write("Run ID:", data.get("run_id"))
            st.write("Run Name:", data.get("run_name"))
            st.write("Version:", data.get("version"))

    except Exception:
        st.error("Could not fetch model info")

    try:
        health_res = requests.get(f"{BACKEND_URL}/health", timeout=10)
        health = health_res.json()
        st.subheader("🔄 Backend Refresh")
        st.write("Model loaded:", health.get("model_loaded"))
        st.write("Current serving version:", health.get("model_version"))
        st.write("Last refresh check:", health.get("model_last_refresh_at"))
    except Exception:
        st.caption("Backend refresh status unavailable")
