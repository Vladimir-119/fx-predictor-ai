"""Загрузка конкретной версии один раз и inference без запросов к Registry."""

from datetime import datetime

import asyncpg
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from src.config import Settings
from src.ml.features import FEATURE_COLUMNS, MIN_HISTORY, latest_features
from src.schemas import ForecastRequest, ForecastResponse, ModelStatus
from src.services.rates import expected_latest_date, load_history, next_publication_date


class LoadedModel(BaseModel):
    """Хранить модель и происхождение в состоянии процесса приложения."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    predictor: mlflow.pyfunc.PyFuncModel
    status: ModelStatus


class ForecastUnavailableError(RuntimeError):
    """Данные или модель не позволяют обслужить прогноз."""


class InvalidForecastDateError(ValueError):
    """Запрошенная дата не совпадает со следующим днём публикации."""


def load_model(settings: Settings) -> LoadedModel:
    """Сначала разрешить alias, затем загрузить именно полученный номер."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    selected = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    predictor = mlflow.pyfunc.load_model(f"models:/{settings.model_name}/{selected.version}")
    metadata = predictor.metadata.metadata or {}
    if metadata.get("pair") != "USD/EUR" or metadata.get("target") != "next_rate_change":
        raise ValueError("Registered model has an incompatible pair or target")
    if metadata.get("features") != FEATURE_COLUMNS:
        raise ValueError("Registered model has an incompatible feature contract")
    return LoadedModel(
        predictor=predictor,
        status=ModelStatus(
            name=settings.model_name,
            version=str(selected.version),
            run_id=selected.run_id,
            alias_at_startup=settings.model_alias,
        ),
    )


def predict_rate(model: LoadedModel, history: pd.DataFrame) -> float:
    """Выполнить CPU-inference и проверить конечный положительный результат."""
    inputs = latest_features(history)
    changes = np.asarray(model.predictor.predict(inputs), dtype=float).reshape(-1)
    if len(changes) != 1:
        raise ForecastUnavailableError("Model must return one change per observation")
    predicted = float(history.rate.iloc[-1] + changes[0])
    if not np.isfinite(predicted) or predicted <= 0:
        raise ForecastUnavailableError("Model returned an invalid rate")
    return predicted


async def forecast(
    request: ForecastRequest,
    model: LoadedModel,
    pool: asyncpg.Pool | asyncpg.Connection,
    now: datetime,
    *,
    save: bool = True,
) -> ForecastResponse:
    """Построить прогноз из актуальной истории и сохранить его происхождение."""
    history = await load_history(pool, now.date())
    if history.empty or history.date.iloc[-1].date() < expected_latest_date(now):
        raise ForecastUnavailableError("Rates are missing or stale; run make update-rates")
    if len(history) < MIN_HISTORY:
        raise ForecastUnavailableError(
            f"At least {MIN_HISTORY} published observations are required"
        )
    as_of = history.date.iloc[-1].date()
    target_date = next_publication_date(as_of)
    if request.forecast_date is not None and request.forecast_date != target_date:
        raise InvalidForecastDateError(f"Only the next ECB publication is supported: {target_date}")
    predicted = await run_in_threadpool(predict_rate, model, history)
    response = ForecastResponse(
        as_of=as_of,
        forecast_date=target_date,
        last_rate=float(history.rate.iloc[-1]),
        predicted_rate=predicted,
        model=model.status,
    )
    if not save:
        return response
    await pool.execute(
        """INSERT INTO predictions
        (source, pair, as_of, forecast_date, predicted_rate, model_name, model_version)
        VALUES ($1, $2, $3, $4, $5, $6, $7)""",
        response.source,
        response.pair,
        response.as_of,
        response.forecast_date,
        response.predicted_rate,
        response.model.name,
        response.model.version,
    )
    return response
