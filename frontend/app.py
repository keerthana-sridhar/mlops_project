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
    ### 🧪 How to Use

    1. Go to **Home**
    2. Upload blood smear images
    3. Click **Run Analysis**
    4. View predictions and download results

    ### 📊 Output

    - Prediction: Parasitized / Uninfected
    - Confidence Score

    ### ⚠️ Notes

    - Supported formats: JPG, PNG, ZIP (containing images)
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

    # -------- PROCESS INPUTS -------- #
    def process_file(file, name_override=None):
        filename = name_override or file.name

        # Size check
        size_mb = len(file.getvalue()) / (1024 * 1024)
        if size_mb > MAX_SIZE_MB:
            failed_files.append({
                "filename": filename,
                "reason": "File too large"
            })
            return

        try:
            image = Image.open(file)
            image.verify()  # 🔥 detects corrupt / fake images

            image = Image.open(file).convert("RGB")  # reopen after verify

            valid_files.append((filename, file, image))

        except Exception:
            failed_files.append({
                "filename": filename,
                "reason": "Corrupt file"
            })

    if uploaded_files:

        # Handle ZIP + normal files
        for file in uploaded_files:

            if file.name.endswith(".zip"):
                try:
                    zip_bytes = io.BytesIO(file.getvalue())
                    with zipfile.ZipFile(zip_bytes) as z:
                        for name in z.namelist():
                            if name.lower().endswith(("png", "jpg", "jpeg")):

                                # Skip macOS metadata files ONLY
                                if "__MACOSX" in name or os.path.basename(name).startswith("._"):
                                    continue
                                extracted = io.BytesIO(z.read(name))
                                extracted.name = name
                                process_file(extracted, name_override=name)
                except Exception:
                    failed_files.append({
                        "filename": file.name,
                        "reason": "Invalid ZIP file"
                    })
            else:
                process_file(file)

        # -------- PREVIEW -------- #
        st.subheader("Preview")

        if valid_files:
            cols = st.columns(min(4, len(valid_files)))

            for i, (_, _, image) in enumerate(valid_files):
                cols[i % 4].image(image, use_container_width=True)
        else:
            st.warning("No valid images to preview (all files are corrupt or invalid)")

        # Show failed files early
        if failed_files:
            st.warning(f"{len(failed_files)} files skipped ❌")

        # -------- RUN ANALYSIS -------- #
        if st.button("🔍 Run Analysis"):

            if not valid_files:
                st.error("No valid images to process")
                st.stop()

            results = []

            with st.spinner(f"Processing {len(valid_files)} images..."):

                for filename, file, _ in valid_files:
                    try:
                        mime_type = mimetypes.guess_type(filename)[0]

                        if mime_type is None:
                            mime_type = "application/octet-stream"

                        files = {
                            "file": (filename, file.getvalue(), mime_type)
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
                                "status": "failed"
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
                    st.error(f"{r['filename']}: {r['alert']} — {r['message']}")
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
    st.image("frontend/assets/pipeline.png")  # your diagram

    st.caption("Note: Monitoring and drift detection are planned extensions.")

    st.subheader("🔁 DVC Pipeline DAG")
    st.image("frontend/assets/dvc_dag.png")  # screenshot from dvc dag

    st.subheader("📟 Pipeline Status")
    try:
        res = requests.get(f"{BACKEND_URL}/pipeline/status")
        data = res.json()

        if data.get("clean"):
            st.success("Pipeline is up-to-date ✅")
        else:
            st.warning("Pipeline needs update ⚠️")

        st.code(data.get("dvc_status", "No output"), language="bash")

    except Exception:
        st.error("Could not fetch pipeline status")

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
