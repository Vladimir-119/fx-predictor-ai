"""Bootstrap не меняет champion и регистрирует совместимый baseline только для пустого стенда."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from scripts import bootstrap_observability as bootstrap
from src.ml.features import FEATURE_COLUMNS, FEATURE_SET


def test_bootstrap_keeps_existing_alias_and_propagates_registry_errors(settings, monkeypatch):
    settings.model_alias = "champion"
    client = MagicMock()
    client.get_registered_model.return_value = SimpleNamespace(aliases={"champion": "7"})
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7")
    load_rates = MagicMock()
    monkeypatch.setattr(bootstrap, "MlflowClient", MagicMock(return_value=client))
    monkeypatch.setattr(bootstrap.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(bootstrap, "load_rates", load_rates)
    assert bootstrap.bootstrap(settings) == "7"
    load_rates.assert_not_called()
    client.set_registered_model_alias.assert_not_called()
    client.get_registered_model.side_effect = MlflowException("unavailable")
    with pytest.raises(MlflowException, match="unavailable"):
        bootstrap.bootstrap(settings)
    load_rates.assert_not_called()


def test_bootstrap_fits_real_features_and_registers_matching_contract(settings, monkeypatch):
    settings.model_alias = "champion"
    client = MagicMock()
    client.get_registered_model.side_effect = [
        MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST),
        SimpleNamespace(aliases={}),
    ]
    rates = pd.DataFrame(
        {"date": pd.bdate_range("2025-01-01", periods=30), "rate": np.linspace(0.8, 0.9, 30)}
    )
    monkeypatch.setattr(bootstrap, "MlflowClient", MagicMock(return_value=client))
    monkeypatch.setattr(bootstrap, "load_rates", lambda: rates)
    monkeypatch.setattr(bootstrap, "read_manifest", lambda: {"sha256": "checked-by-load-rates"})
    for name in ("set_tracking_uri", "set_experiment", "log_params", "log_input", "log_dict"):
        monkeypatch.setattr(bootstrap.mlflow, name, MagicMock())
    monkeypatch.setattr(bootstrap.mlflow, "start_run", lambda **_kwargs: nullcontext())
    logged = MagicMock(return_value=SimpleNamespace(model_uri="models:/logged-model"))
    monkeypatch.setattr(bootstrap.mlflow.sklearn, "log_model", logged)
    monkeypatch.setattr(
        bootstrap.mlflow,
        "register_model",
        MagicMock(return_value=SimpleNamespace(version="1")),
    )
    assert bootstrap.bootstrap(settings) == "1"
    model = logged.call_args.args[0]
    assert model.n_features_in_ == len(FEATURE_COLUMNS)
    assert model.feature_names_in_.tolist() == FEATURE_COLUMNS
    np.testing.assert_array_equal(model.predict(logged.call_args.kwargs["input_example"]), 0)
    assert logged.call_args.kwargs["metadata"] == {
        "pair": "USD/EUR",
        "target": "next_rate_change",
        "features": FEATURE_COLUMNS,
        "feature_set": FEATURE_SET,
    }
    client.set_registered_model_alias.assert_called_once_with(settings.model_name, "champion", "1")
