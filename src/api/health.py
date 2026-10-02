"""Health-эндпоинты приложения.

Два уровня (стандарт liveness/readiness):

- `GET /healthz` — быстрый liveness: процесс жив и отвечает.
  Без внешних вызовов. Используется Docker-healthcheck'ом и оркестраторами.
- `GET /api/v1/health` — подробный отчёт: статус приложения и каждой
  зависимости, время проверки, метаданные (версия, среда).
  HTTP 200, только если все зависимости здоровы, иначе 503.
"""

import logging

from fastapi import APIRouter, Request, Response, status

from src.schemas import HealthReport, LivenessResponse, ReportStatus
from src.services import health as health_service

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/healthz", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Вернуть статус живости процесса."""
    return LivenessResponse()


@router.get(
    "/api/v1/health",
    response_model=HealthReport,
    responses={503: {"model": HealthReport, "description": "PostgreSQL is unavailable"}},
)
async def health(request: Request, response: Response) -> HealthReport:
    """Вернуть подробный health-отчёт по приложению и зависимостям."""
    report = await health_service.build_report(
        request.app.state.db_pool, request.app.state.settings
    )
    if report.status is not ReportStatus.ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        log.warning("health_degraded")
    return report
