# User Manual

## Purpose

This application helps clinicians and evaluators classify malaria cell images and monitor the deployed AI model.

## Step 1: Start the System

Create `.env` from `.env.example`, then use Docker Compose to start the containers. After startup, open:

- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`
- Grafana: `http://localhost:3001`
- Airflow: `http://localhost:8080`

## Step 2: Run Predictions

1. Open the **Home** page.
2. Upload one or more `.jpg`, `.jpeg`, `.png`, or `.zip` files.
3. Click **Run Analysis**.
4. Review the prediction table and the alert section.

## Step 3: Understand Review Queues

- **Normal accepted samples** may be stored in `data/feedback/unlabeled`.
- **OOD or uncertain samples** are stored in `data/feedback/unlabeled_ood`.

These files require manual physician review.

## Step 4: Manual Labelling

After physician review, move files into:

- `data/feedback/labelled/Parasitized`
- `data/feedback/labelled/Uninfected`

Only reviewed, medically verified images should be moved into the labelled folders.

## Step 5: Retraining

There are two retraining paths:

### Airflow Feedback Retraining

Use the **Pipeline** page and click **Trigger Retraining DAG**.

This runs the operational Airflow DAG that:

- checks for new labelled feedback
- processes feedback
- fine-tunes the model
- evaluates the model
- promotes the model in MLflow if evaluation passes

### Full Reproducible DVC Pipeline

Run:

```bash
docker compose exec backend dvc repro
```

This reruns the complete tracked DVC pipeline.

## Step 6: Monitoring

- Use the frontend sidebar for quick metrics.
- Use Grafana for Prometheus visualizations.
- Use the **Pipeline** page to see DVC status, feedback queue size, and retraining status.
