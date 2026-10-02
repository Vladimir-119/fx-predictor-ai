"""Управление асинхронным пулом PostgreSQL."""

import logging

import asyncpg

from src.config import Settings

log = logging.getLogger(__name__)


async def create_db_pool(settings: Settings) -> asyncpg.Pool:
    """Создать ленивый пул: отсутствие БД не мешает liveness-проверке."""
    pool = await asyncpg.create_pool(
        settings.database_url.get_secret_value(),
        min_size=0,
        max_size=settings.db_pool_max_size,
        timeout=settings.health_timeout_seconds,
        command_timeout=settings.health_timeout_seconds,
    )
    log.info("database_pool_created")
    return pool


async def close_db_pool(pool: asyncpg.Pool) -> None:
    """Закрыть пул и все его соединения."""
    await pool.close()
    log.info("database_pool_closed")
