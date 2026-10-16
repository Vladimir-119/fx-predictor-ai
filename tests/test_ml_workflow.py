"""Настоящие маленькие модели и Registry без сети и внешней БД."""

import hashlib
import os
import uuid

import mlflow
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from mlflow import MlflowClient

from src.ml import eda, train
from src.ml.features import make_training_frame
from src.services.prediction import load_model, predict_rate


def test_real_experiments_registry_and_alias_reload(settings, tmp_path, monkeypatch):
    settings.mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TEST_URI", f"sqlite:///{tmp_path / 'tracking.db'}"
    )
    settings.mlflow_experiment = f"test-workflow-{uuid.uuid4().hex[:8]}"
    settings.model_alias = "champion"
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    if settings.mlflow_tracking_uri.startswith("sqlite:"):
        experiment = client.create_experiment(
            settings.mlflow_experiment, artifact_location=(tmp_path / "artifacts").as_uri()
        )
    else:
        experiment = client.create_experiment(settings.mlflow_experiment)
    rates = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=160),
            "rate": 0.9 + np.sin(np.arange(160) / 8) * 0.01,
        }
    )
    snapshot = tmp_path / "test-rates.csv"
    snapshot.write_text(rates.to_csv(index=False), encoding="utf-8")
    manifest = {
        "source": snapshot.as_uri(),
        "dataset_version": "synthetic-test-fixture",
        "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(eda, "RAW_PATH", snapshot)
    monkeypatch.setattr(eda, "read_manifest", lambda: manifest)
    monkeypatch.setattr(train, "read_manifest", lambda: manifest)
    with pytest.raises(ValueError, match="Unknown model"):
        train.create_model("unknown")
    figures = eda.make_plots(rates)
    eda_id = eda.log_eda(settings, rates, figures)
    plt.close("all")
    result = train.train(settings, make_training_frame(rates))
    assert result["test_mae"] >= 0
    versions = client.search_model_versions(f"name = '{settings.model_name}'")
    assert len(versions) == 4
    runs = client.search_runs([experiment])
    assert len(runs) == 6
    assert client.get_run(eda_id).inputs.dataset_inputs
    assert all(run.inputs.dataset_inputs for run in runs)
    assert any("cv_mae" in run.data.metrics for run in runs)
    loaded = load_model(settings)
    old_version = loaded.status.version
    assert old_version == result["champion_version"]
    assert np.isfinite(predict_rate(loaded, rates.tail(60)))
    client.set_registered_model_alias(
        settings.model_name, "champion", str(result["challenger_version"])
    )
    assert loaded.status.version == old_version
    refreshed = load_model(settings)
    assert refreshed.status.version == result["challenger_version"]
    assert refreshed.status.run_id != loaded.status.run_id
    assert np.isfinite(predict_rate(refreshed, rates.tail(60)))
    client.set_registered_model_alias(settings.model_name, "champion", old_version)
