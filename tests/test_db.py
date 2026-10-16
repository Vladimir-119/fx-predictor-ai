"""Пул ленивый, асинхронный и закрывается даже после ошибки."""

from unittest.mock import AsyncMock

import pytest
from asgi_lifespan import LifespanManager

from src import db
from src.app import create_app


async def test_pool_is_lazy_and_lifespan_closes_it_on_success_and_failure(
    settings, pool, monkeypatch
):
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

    application = create_app(settings)
    async with LifespanManager(application):
        assert application.state.db_pool is pool
        pool.close.assert_not_awaited()
    pool.close.assert_awaited_once()

    pool.close.reset_mock()
    with pytest.raises(RuntimeError, match="test failure"):
        async with application.router.lifespan_context(application):
            raise RuntimeError("test failure")
    pool.close.assert_awaited_once()
