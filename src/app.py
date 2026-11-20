"""Точка сборки FastAPI: uv run --extra web uvicorn src.app:app --reload."""

import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src import config, db, logging_config, messaging
from src.api import health, hello, process, tasks, version
from src.services import prediction

log = logging.getLogger(__name__)
MAX_REQUEST_BODY_BYTES = 64 * 1024


class RequestBodyLimitMiddleware:
    """Ограничить фактическое тело до разбора JSON, включая chunked-запросы."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_REQUEST_BODY_BYTES:
                response = JSONResponse({"detail": "Request body exceeds 64 KiB"}, status_code=413)
                await response(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        pending = True

        async def receive_body() -> Message:
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, receive_body, send)


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
        app.state.task_broker = None
        if settings.tasks_enabled:
            broker = messaging.create_broker(settings)
            try:
                await messaging.connect_broker(broker, settings)
            except (OSError, TimeoutError, ConnectionError):
                await broker.stop()
                log.warning("task_broker_unavailable")
            else:
                app.state.task_broker = broker
        app.state.forecast_model = None
        if settings.model_alias:
            app.state.forecast_model = await run_in_threadpool(prediction.load_model, settings)
            log.info("model_loaded", extra={"model": app.state.forecast_model.status.model_dump()})
        yield
    finally:
        if app.state.task_broker is not None:
            await app.state.task_broker.stop()
        await db.close_db_pool(app.state.db_pool)
        log.info("application_stopped")


def create_app(settings: config.Settings | None = None) -> FastAPI:
    """Собрать приложение; тесты могут передать собственные настройки."""
    settings = settings or config.get_settings()
    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
    app.add_middleware(RequestBodyLimitMiddleware)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts, www_redirect=False
    )
    app.state.settings = settings
    app.state.forecast_model = None
    app.state.task_broker = None

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
    app.include_router(process.router)
    app.include_router(tasks.router)
    return app


app = create_app()
