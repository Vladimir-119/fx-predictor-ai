"""Проверка версии БД, таймаута и корректной обработки ошибок."""

import asyncio

import pytest
from asyncpg import PostgresError

from src.schemas import HealthStatus
from src.services import health as health_service


async def test_database_health_bounds_waiting_handles_errors_and_preserves_cancellation(
    pool, settings
):
    pool.fetchval.side_effect = PostgresError("private")
    dependency = await health_service.check_postgres(pool, settings)
    assert dependency.status is HealthStatus.unavailable
    assert dependency.version is None
    assert dependency.response_time_ms >= 0
    assert dependency.detail == "PostgreSQL is unavailable or the check timed out"
    assert "private" not in dependency.model_dump_json()

    pool.fetchval.side_effect = None
    pool.fetchval.return_value = None
    dependency = await health_service.check_postgres(pool, settings)
    assert dependency.status is HealthStatus.unavailable
    assert dependency.version is None
    assert dependency.detail == "PostgreSQL returned an invalid version"

    cancelled = asyncio.Event()

    async def stalled_query(query):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    pool.fetchval.side_effect = stalled_query
    settings.health_timeout_seconds = 0.02
    dependency = await asyncio.wait_for(health_service.check_postgres(pool, settings), timeout=1)
    assert dependency.status is HealthStatus.unavailable
    assert dependency.response_time_ms >= 10
    assert cancelled.is_set()

    async def delayed_query(query):
        await asyncio.sleep(0.02)
        return "16.10"

    settings.health_timeout_seconds = 0.2
    pool.fetchval.side_effect = delayed_query
    dependency = await health_service.check_postgres(pool, settings)
    assert dependency.status is HealthStatus.healthy
    assert dependency.response_time_ms >= 10

    pool.fetchval.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await health_service.check_postgres(pool, settings)
