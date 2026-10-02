"""Точка сборки FastAPI: uv run --extra web uvicorn src.app:app --reload."""

import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.responses import Response

from src import config, db, logging_config
from src.api import health, hello, version

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Создать пул БД при старте, закрыть при остановке."""
    settings = app.state.settings
    logging_config.setup_logging(settings.log_level)
    app.state.db_pool = await db.create_db_pool(settings)
    log.info(
        "application_started",
        extra={"app": settings.app_name, "version": settings.version},
    )
    try:
        yield
    finally:
        await db.close_db_pool(app.state.db_pool)
        log.info("application_stopped")


def create_app(settings: config.Settings | None = None) -> FastAPI:
    """Собрать приложение; тесты могут передать собственные настройки."""
    settings = settings or config.get_settings()
    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
    app.state.settings = settings

    @app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Связать логи запроса и вернуть X-Request-ID в ответе."""
        request_id = uuid.uuid4().hex
        token = logging_config.REQUEST_ID.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": (time.perf_counter() - start) * 1000,
                },
            )
            raise
        else:
            log.info(
                "request_completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": (time.perf_counter() - start) * 1000,
                },
            )
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            logging_config.REQUEST_ID.reset(token)

    app.include_router(hello.router)
    app.include_router(health.router)
    app.include_router(version.router)
    return app


app = create_app()
