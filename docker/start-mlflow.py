"""Запустить Tracking Server с PostgreSQL и проксированием S3-артефактов."""

import os
from urllib.parse import quote

user = quote(os.environ["POSTGRES_USER"], safe="")
password = quote(os.environ["POSTGRES_PASSWORD"], safe="")
database = quote(os.environ["MLFLOW_DB"], safe="")
os.execvp(
    "uv",
    [
        "uv",
        "run",
        "--no-project",
        "mlflow",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--workers",
        "1",
        "--backend-store-uri",
        f"postgresql+psycopg2://{user}:{password}@postgres:5432/{database}",
        "--artifacts-destination",
        f"s3://{os.environ['MLFLOW_BUCKET']}",
        "--allowed-hosts",
        "mlflow:*,localhost:*,127.0.0.1:*",
    ],
)
