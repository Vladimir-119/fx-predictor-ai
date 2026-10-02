"""Pydantic-контракты API приложения."""

from enum import StrEnum

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
