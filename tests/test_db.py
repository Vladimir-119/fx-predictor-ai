"""Пул ленивый, асинхронный и закрывается даже после ошибки."""

from unittest.mock import AsyncMock

import pytest
from asgi_lifespan import LifespanManager

from src import db


async def test_create_pool_is_lazy_and_uses_configured_limits(settings, pool, monkeypatch):
    factory = AsyncMock(return_value=pool)
    monkeypatch.setattr(db.asyncpg, "create_pool", factory)
    result = await db.create_db_pool(settings)
    assert result is pool
    factory.assert_awaited_once_with(
        settings.database_url.get_secret_value(),
        min_size=0,
        max_size=settings.db_pool_max_size,
        timeout=settings.health_timeout_seconds,
        command_timeout=settings.health_timeout_seconds,
    )


async def test_lifespan_closes_pool_on_normal_shutdown(application, pool):
    async with LifespanManager(application):
        assert application.state.db_pool is pool
        pool.close.assert_not_awaited()
    pool.close.assert_awaited_once()


async def test_lifespan_closes_pool_after_application_error(application, pool):
    with pytest.raises(RuntimeError, match="test failure"):
        async with application.router.lifespan_context(application):
            raise RuntimeError("test failure")
    pool.close.assert_awaited_once()
