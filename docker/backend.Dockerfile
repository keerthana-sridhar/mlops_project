FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/project

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/backend-requirements.txt /tmp/backend-requirements.txt

RUN pip install -r /tmp/backend-requirements.txt && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
