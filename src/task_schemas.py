"""Сообщения очереди и публичные ответы API задач."""

from datetime import UTC, date, datetime
from itertools import pairwise
from typing import Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from src.schemas import ForecastRequest, ForecastResponse


class HistoricalRate(BaseModel):
    """Один известный курс: EUR за один USD."""

    model_config = ConfigDict(extra="forbid")

    date: date
    rate: float = Field(gt=0, allow_inf_nan=False)


class TaskRequest(ForecastRequest):
    """Без history используется свежая БД; history нужен для пакетного пересчёта."""

    task_id: UUID = Field(default_factory=uuid4)
    history: list[HistoricalRate] | None = Field(default=None, min_length=20, max_length=60)

    @field_validator("history")
    @classmethod
    def validate_history(cls, history: list[HistoricalRate] | None) -> list[HistoricalRate] | None:
        """Отклонить некорректную историю до публикации задачи в брокер."""
        if history:
            if any(current.date <= previous.date for previous, current in pairwise(history)):
                raise ValueError("History dates must be unique and strictly increasing")
            if history[-1].date > datetime.now(UTC).date():
                raise ValueError("History cannot contain future observations")
        return history


class TaskMessage(BaseModel):
    """Неизменяемое сообщение с временем отправки для измерения задержки."""

    model_config = ConfigDict(extra="forbid")

    task: TaskRequest
    submitted_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskAccepted(BaseModel):
    """Подтверждение приёма; task_id можно использовать повторно при сетевом сбое."""

    task_id: UUID
    status: str = "QUEUED"


class TaskResult(BaseModel):
    """Состояние задачи и версия модели, фактически выполнившей inference."""

    task_id: UUID
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]
    attempts: int
    result: ForecastResponse | None = None
    error: str | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
