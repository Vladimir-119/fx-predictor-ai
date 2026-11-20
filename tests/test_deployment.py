"""Проверки безопасного запуска служебных процессов."""

import os
import runpy
from importlib import reload
from urllib.parse import quote

import pytest


def test_local_commands_disable_mlflow_usage_telemetry(monkeypatch):
    """Обычный import пакета задаёт privacy-default до использования MLflow."""
    import src

    monkeypatch.delenv("MLFLOW_DISABLE_TELEMETRY", raising=False)
    reload(src)
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"


def test_mlflow_database_password_is_not_exposed_in_process_arguments(monkeypatch):
    """DSN передаётся через env, а специальные символы пароля кодируются."""
    password = "private-pass@word:/?#"
    monkeypatch.setenv("POSTGRES_USER", "fx_predictor")
    monkeypatch.setenv("POSTGRES_PASSWORD", password)
    monkeypatch.setenv("MLFLOW_DB", "mlflow")
    monkeypatch.setenv("MLFLOW_BUCKET", "mlflow-artifacts")
    # Обеспечить восстановление env после выполнения скрипта через runpy.
    monkeypatch.setenv("MLFLOW_BACKEND_STORE_URI", "")
    command = []

    def capture_exec(program, arguments):
        assert program == "uv"
        command.extend(arguments)
        raise SystemExit

    monkeypatch.setattr(os, "execvp", capture_exec)
    with pytest.raises(SystemExit):
        runpy.run_path("docker/start-mlflow.py", run_name="__main__")

    assert "--backend-store-uri" not in command
    assert password not in " ".join(command)
    assert quote(password, safe="") not in " ".join(command)
    assert os.environ["MLFLOW_BACKEND_STORE_URI"] == (
        f"postgresql+psycopg2://fx_predictor:{quote(password, safe='')}@postgres:5432/mlflow"
    )
