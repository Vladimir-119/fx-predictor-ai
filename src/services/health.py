"""Проверка приложения и реального соединения с PostgreSQL."""

import asyncio
import logging
from time import perf_counter

from asyncpg import Pool, PostgresError

from src.config import Settings
from src.schemas import DependencyHealth, HealthReport, HealthStatus, ReportStatus

log = logging.getLogger(__name__)


def check_application(settings: Settings) -> DependencyHealth:
    """Вернуть версию приложения и время получения метаданных."""
    start = perf_counter()
    version = settings.version
    return DependencyHealth(
        status=HealthStatus.healthy,
        version=version,
        response_time_ms=(perf_counter() - start) * 1000,
        detail=settings.app_name,
    )


async def check_postgres(pool: Pool, settings: Settings) -> DependencyHealth:
    """Измерить получение соединения, запрос версии и возврат соединения."""
    start = perf_counter()
    try:
        version = await asyncio.wait_for(
            pool.fetchval("SELECT current_setting('server_version')"),
            timeout=settings.health_timeout_seconds,
        )
    except (OSError, PostgresError, TimeoutError) as exc:
        log.warning("postgres_unavailable", extra={"error_type": type(exc).__name__})
        return DependencyHealth(
            status=HealthStatus.unavailable,
            response_time_ms=(perf_counter() - start) * 1000,
            detail="PostgreSQL is unavailable or the check timed out",
        )

    if not isinstance(version, str) or not version:
        log.error("postgres_invalid_version")
        return DependencyHealth(
            status=HealthStatus.unavailable,
            response_time_ms=(perf_counter() - start) * 1000,
            detail="PostgreSQL returned an invalid version",
        )

    return DependencyHealth(
        status=HealthStatus.healthy,
        version=version,
        response_time_ms=(perf_counter() - start) * 1000,
    )


async def build_report(pool: Pool, settings: Settings) -> HealthReport:
    """Собрать статусы, версии и время проверки всех подключённых сервисов."""
    dependencies = {
        "application": check_application(settings),
        "postgres": await check_postgres(pool, settings),
    }
    healthy = all(dep.status is HealthStatus.healthy for dep in dependencies.values())
    return HealthReport(
        status=ReportStatus.ok if healthy else ReportStatus.degraded,
        environment=settings.environment,
        dependencies=dependencies,
    )
