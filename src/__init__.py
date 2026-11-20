"""Приложение fx-predictor-ai."""

import os

# Локальные команды, как Docker и CI, не отправляют usage-телеметрию MLflow.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
