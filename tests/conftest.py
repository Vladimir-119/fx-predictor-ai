"""Изолированные тесты: вместо внешней БД используется AsyncMock."""

from unittest.mock import AsyncMock

import pytest
from asgi_lifespan import LifespanManager
from asyncpg import Pool
from httpx import ASGITransport, AsyncClient

from src import db
from src.app import create_app
from src.config import Settings


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        database_url="postgresql://test:test@127.0.0.1:59999/fx_predictor_ai",
        log_level="INFO",
        environment="test",
        health_timeout_seconds=0.2,
        db_pool_max_size=5,
    )


@pytest.fixture
def pool():
    mock = AsyncMock(spec=Pool)
    mock.fetchval.return_value = "16.10"
    return mock


@pytest.fixture
def application(settings, pool, monkeypatch):
    monkeypatch.setattr(db, "create_db_pool", AsyncMock(return_value=pool))
    return create_app(settings)


@pytest.fixture
async def client(application):
    async with LifespanManager(application):
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as http:
            yield http
