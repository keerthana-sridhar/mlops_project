# Reproducibility and Portability

## Reproducibility model used in this repo

This project is set up so that the full DVC pipeline can start from a local dataset archive instead of a DVC remote.

The expected input is:

```text
data/raw/raw_zipped.zip
```

The first DVC stage, `prepare_raw_data`, unpacks that archive into:

```text
data/raw/cell_images/
```

After that, the rest of the tracked pipeline runs normally:

1. `eda`
2. `preprocess`
3. `resize`
4. `train`
5. `evaluate`

## What to commit

Commit:

- source code
- `docker-compose.yaml`
- `dvc.yaml`
- `dvc.lock`
- `params.yaml`
- `.env.example`
- documentation
- `models/cnn.pth`

Do not commit:

- `.env`
- `.dvc/config.local`
- `.dvc/cache/`
- `mlruns/`
- `shared/mlflow/`
- `logs/`
- `airflow/logs/`
- extracted data under `data/raw/cell_images/`
- processed data under `data/processed/`

## Important GitHub limitation

If `data/raw/raw_zipped.zip` is larger than GitHub's regular file limit, it should not be pushed as a normal Git object.

Use one of these instead:

- Git LFS
- GitHub Release asset
- external file share, with instructions to place the file at `data/raw/raw_zipped.zip`

## How another machine reproduces the pipeline

1. Clone the repository.
2. Create `.env` from `.env.example`.
3. Ensure `data/raw/raw_zipped.zip` is present.
4. Start the Docker stack with `docker compose up --build`.
5. Run `docker compose exec backend dvc repro`.

For a non-Docker local run, step 5 can be replaced with:

```bash
dvc repro
```

## MLflow portability

MLflow runtime data is stored under:

```text
shared/mlflow/
```

That directory is intentionally ignored by Git.
It is not required for the app to boot, because the backend falls back to `models/cnn.pth` when no Production model exists in MLflow.

## Secrets

Keep secrets out of Git:

- use `.env` for container environment variables
- use `.dvc/config.local` only if you ever add a private DVC remote later
