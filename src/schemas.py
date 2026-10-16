"""Pydantic-контракты API приложения."""

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    """Ответ проверки живости процесса."""

    status: str = "ok"


class VersionResponse(BaseModel):
    """Название и версия установленной сборки приложения."""

    app: str
    version: str


class HelloResponse(VersionResponse):
    """Приветствие из шаблона преподавателя."""

    message: str


class HealthStatus(StrEnum):
    """Статус отдельной зависимости."""

    healthy = "healthy"
    unavailable = "unavailable"


class ReportStatus(StrEnum):
    """Общий статус приложения в health-отчёте."""

    ok = "ok"
    degraded = "degraded"


class DependencyHealth(BaseModel):
    """Версия компонента и время его полной проверки в миллисекундах."""

    status: HealthStatus
    version: str | None = None
    response_time_ms: float = Field(ge=0)
    detail: str | None = None


class HealthReport(BaseModel):
    """Сводный health-отчёт приложения."""

    status: ReportStatus
    environment: str
    dependencies: dict[str, DependencyHealth]


class RateObservation(BaseModel):
    """Опубликованное значение одной пары в исходном направлении ECB."""

    source: Literal["ecb"] = "ecb"
    base: str
    quote: str
    rate_type: Literal["reference"] = "reference"
    rate_date: date
    value: float = Field(gt=0, allow_inf_nan=False)


class ForecastRequest(BaseModel):
    """Одна поддерживаемая пара и следующая дата публикации."""

    pair: Literal["USD/EUR"] = "USD/EUR"
    forecast_date: date | None = None


class ModelStatus(BaseModel):
    """Идентифицировать именно загруженную в память версию."""

    name: str
    version: str
    run_id: str
    alias_at_startup: str
    pair: Literal["USD/EUR"] = "USD/EUR"


class ForecastResponse(BaseModel):
    """Прогноз следующего справочного курса с происхождением модели."""

    pair: Literal["USD/EUR"] = "USD/EUR"
    source: Literal["ecb"] = "ecb"
    as_of: date
    forecast_date: date
    last_rate: float = Field(gt=0, allow_inf_nan=False)
    predicted_rate: float = Field(gt=0, allow_inf_nan=False)
    model: ModelStatus


class DisplayRate(BaseModel):
    """Выразить последнюю котировку в единицах валюты за один USD."""

    quote: str
    value: float = Field(gt=0, allow_inf_nan=False)


class RatesResponse(BaseModel):
    """Последняя таблица ECB с явной датой и проверкой свежести."""

    source: Literal["ecb"] = "ecb"
    base: Literal["USD"] = "USD"
    rate_date: date
    stale: bool
    rates: list[DisplayRate]
