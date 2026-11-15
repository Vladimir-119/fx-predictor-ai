"""Контракт API, один startup-load и отзывчивость во время CPU-inference."""

import asyncio
import threading
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.ml.features import FEATURE_COLUMNS
from src.schemas import ForecastRequest, ModelStatus
from src.services import prediction


def loaded_model(predictor=None):
    return prediction.LoadedModel.model_construct(
        predictor=predictor or SimpleNamespace(predict=lambda inputs: np.array([0.001])),
        status=ModelStatus(
            name="fx-predictor-ai-forecaster",
            version="4",
            run_id="run-4",
            alias_at_startup="champion",
        ),
    )


def history_rows():
    return [
        {"rate_date": day.date(), "value": 1.2}
        for day in pd.bdate_range(end="2026-10-07", periods=30)[::-1]
    ]


def test_loader_resolves_alias_before_loading_exact_version_and_checks_contract(
    settings, monkeypatch
):
    registry = MagicMock()
    registry.get_model_version_by_alias.return_value = SimpleNamespace(version="4", run_id="run-4")
    predictor = MagicMock(spec=prediction.mlflow.pyfunc.PyFuncModel)
    metadata = {"pair": "USD/EUR", "target": "next_rate_change", "features": FEATURE_COLUMNS}
    predictor.metadata = SimpleNamespace(metadata=metadata)
    loader = MagicMock(return_value=predictor)
    monkeypatch.setattr(prediction, "MlflowClient", MagicMock(return_value=registry))
    monkeypatch.setattr(prediction.mlflow.pyfunc, "load_model", loader)
    settings.model_alias = "champion"
    model = prediction.load_model(settings)
    loader.assert_called_once_with("models:/fx-predictor-ai-forecaster/4")
    assert model.status.version == "4" and model.status.run_id == "run-4"
    for invalid in ({}, {**metadata, "features": ["unknown"]}):
        predictor.metadata.metadata = invalid
        with pytest.raises(ValueError, match="incompatible"):
            prediction.load_model(settings)


async def test_forecast_uses_available_history_and_rejects_stale_or_wrong_horizon(pool):
    now = datetime(2026, 10, 8, 12, tzinfo=UTC)
    pool.fetch.return_value = history_rows()
    model = loaded_model()
    result = await prediction.forecast(ForecastRequest(), model, pool, now)
    assert result.as_of == date(2026, 10, 7)
    assert result.forecast_date == date(2026, 10, 8)
    assert result.predicted_rate == pytest.approx(1 / 1.2 + 0.001)
    assert pool.fetch.call_args.args[1] == now.date()
    assert pool.execute.call_args.args[-1] == "4"
    pool.execute.reset_mock()
    await prediction.forecast(ForecastRequest(), model, pool, now, save=False)
    pool.execute.assert_not_awaited()
    with pytest.raises(prediction.InvalidForecastDateError, match="next ECB"):
        await prediction.forecast(
            ForecastRequest(forecast_date=date(2026, 10, 9)), model, pool, now
        )
    pool.fetch.return_value = []
    with pytest.raises(prediction.ForecastUnavailableError, match="missing"):
        await prediction.forecast(ForecastRequest(), model, pool, now)
    pool.fetch.return_value = history_rows()[:10]
    with pytest.raises(prediction.ForecastUnavailableError, match="20"):
        await prediction.forecast(ForecastRequest(), model, pool, now)
    history = pd.DataFrame({"date": pd.bdate_range(end="2026-10-07", periods=30), "rate": 0.8})
    for value in (np.nan, -999):
        invalid = loaded_model(SimpleNamespace(predict=lambda inputs: np.array([value])))
        with pytest.raises(prediction.ForecastUnavailableError, match="invalid"):
            prediction.predict_rate(invalid, history)
    with pytest.raises(prediction.ForecastUnavailableError, match="one change"):
        prediction.predict_rate(
            loaded_model(SimpleNamespace(predict=lambda inputs: [1, 2])), history
        )


async def test_model_loads_once_and_liveness_is_responsive_during_inference(
    application, settings, pool, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()

    def slow_predict(inputs):
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("Test did not release inference")
        return np.array([0.0])

    model = loaded_model(SimpleNamespace(predict=slow_predict))
    loader = MagicMock(return_value=model)
    monkeypatch.setattr(prediction, "load_model", loader)
    monkeypatch.setattr(prediction, "expected_latest_date", lambda now: date(2026, 10, 7))
    settings.model_alias = "champion"
    pool.fetch.return_value = history_rows()
    async with LifespanManager(application):
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as http:
            task = asyncio.create_task(http.post("/process", json={}))
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            try:
                assert (await asyncio.wait_for(http.get("/healthz"), timeout=1)).status_code == 200
            finally:
                release.set()
            response = await task
            assert response.status_code == 200
            assert (await http.get("/api/v1/model")).json()["version"] == "4"
            assert (await http.post("/process", json={})).status_code == 200
    loader.assert_called_once()


async def test_process_errors_are_clear_and_startup_failure_closes_pool(
    client, application, pool, settings, monkeypatch
):
    assert (await client.post("/process", json={})).status_code == 503
    assert (await client.get("/api/v1/model")).status_code == 503
    assert (await client.post("/process", json={"pair": "USD/JPY"})).status_code == 422
    pool.fetch.return_value = []
    assert (await client.get("/api/v1/rates")).status_code == 503
    application.state.forecast_model = loaded_model()
    assert (await client.post("/process", json={})).status_code == 503
    pool.fetch.side_effect = OSError("private credentials")
    for method, path in (("post", "/process"), ("get", "/api/v1/rates")):
        response = await getattr(client, method)(path, **({"json": {}} if method == "post" else {}))
        assert response.status_code == 503 and "private" not in response.text
    pool.fetch.side_effect = None
    pool.fetch.return_value = history_rows()
    monkeypatch.setattr(prediction, "expected_latest_date", lambda now: date(2026, 10, 7))
    response = await client.post("/process", json={"forecast_date": "2026-10-09"})
    assert response.status_code == 422
    application.state.forecast_model = loaded_model(
        SimpleNamespace(predict=MagicMock(side_effect=ValueError("private model details")))
    )
    response = await client.post("/process", json={})
    assert response.status_code == 503
    assert response.json() == {"detail": "Model prediction failed"}
    settings.model_alias = "champion"
    monkeypatch.setattr(
        prediction, "load_model", MagicMock(side_effect=ValueError("bad signature"))
    )
    with pytest.raises(ValueError, match="bad signature"):
        async with application.router.lifespan_context(application):
            pass
    assert pool.close.await_count >= 1
