FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/project

COPY docker/frontend-requirements.txt /tmp/frontend-requirements.txt

RUN pip install -r /tmp/frontend-requirements.txt

COPY frontend /opt/project/frontend

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
