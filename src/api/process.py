"""Тонкие маршруты прогноза, курсов и версии загруженной модели."""

from datetime import UTC, datetime

from asyncpg import PostgresError
from fastapi import APIRouter, HTTPException, Request

from src.schemas import ForecastRequest, ForecastResponse, ModelStatus, RatesResponse
from src.services import prediction, rates

router = APIRouter(tags=["Forecasts"])


@router.post("/process", response_model=ForecastResponse)
async def process(body: ForecastRequest, request: Request) -> ForecastResponse:
    """Предсказать следующий опубликованный курс USD/EUR."""
    model = request.app.state.forecast_model
    if model is None:
        raise HTTPException(503, "Model is not loaded; register it and set MODEL_ALIAS=champion")
    try:
        result = await prediction.forecast(
            body, model, request.app.state.db_pool, datetime.now(UTC)
        )
    except prediction.ForecastUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except prediction.InvalidForecastDateError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(503, "Model prediction failed") from exc
    except (PostgresError, OSError, TimeoutError) as exc:
        raise HTTPException(503, "PostgreSQL is unavailable") from exc
    return result


@router.get("/api/v1/model", response_model=ModelStatus)
async def model_status(request: Request) -> ModelStatus:
    """Показать версию в памяти без повторного обращения к MLflow."""
    model = request.app.state.forecast_model
    if model is None:
        raise HTTPException(503, "Model is not loaded")
    return model.status


@router.get("/api/v1/rates", response_model=RatesResponse)
async def current_rates(request: Request) -> RatesResponse:
    """Показать опубликованные курсы и явно отметить устаревшие данные."""
    try:
        result = await rates.latest_rates(request.app.state.db_pool, datetime.now(UTC))
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc
    except (PostgresError, OSError, TimeoutError) as exc:
        raise HTTPException(503, "PostgreSQL is unavailable") from exc
    return result
