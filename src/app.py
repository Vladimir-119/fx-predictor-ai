"""Точка сборки FastAPI: uv run --extra web uvicorn src.app:app --reload."""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from opentelemetry import propagate
from opentelemetry.trace import SpanKind, StatusCode
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src import config, db, logging_config, messaging, telemetry
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
    runtime = telemetry.setup_telemetry(settings)
    app.state.db_pool = None
    try:
        app.state.db_pool = await db.create_db_pool(settings)
        log.info(
            "application_started",
            extra={"app": settings.app_name, "version": settings.version},
        )
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
        log.info("graceful_shutdown_started")
        if app.state.task_broker is not None:
            await app.state.task_broker.stop()
        if app.state.db_pool is not None:
            await db.close_db_pool(app.state.db_pool)
        log.info("application_stopped")
        await asyncio.to_thread(runtime.shutdown)


def create_app(settings: config.Settings | None = None) -> FastAPI:
    """Собрать приложение; тесты могут передать собственные настройки."""
    settings = settings or config.get_settings()
    logging_config.setup_logging(settings.log_level)
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
        with telemetry.get_tracer().start_as_current_span(
            f"{request.method} request",
            context=propagate.extract(dict(request.headers)),
            kind=SpanKind.SERVER,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            code = 500
            try:
                response = await call_next(request)
                code = response.status_code
                response.headers["X-Request-ID"] = request_id
                context = span.get_span_context()
                if context.is_valid:
                    response.headers["X-Trace-ID"] = f"{context.trace_id:032x}"
                if code >= 500:
                    span.set_status(StatusCode.ERROR)
                    span.record_exception(RuntimeError(f"HTTP {code}"))
                return response
            except Exception as exc:
                telemetry.record_error(span, exc)
                log.exception("request_failed", extra={"status_code": 500})
                raise
            finally:
                # Только шаблон маршрута: UUID и произвольные пути не становятся labels.
                route = getattr(request.scope.get("route"), "path", "unmatched")
                span.update_name(f"{request.method} {route}")
                span.set_attributes(
                    {
                        "http.request.method": request.method,
                        "http.route": route,
                        "http.response.status_code": code,
                    }
                )
                duration = time.perf_counter() - start
                method = (
                    request.method
                    if request.method
                    in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
                    else "OTHER"
                )
                attrs = {"route": route, "method": method, "status_code": str(code)}
                telemetry.metrics.api_requests.add(1, attrs)
                telemetry.metrics.api_duration.record(duration, attrs)
                level = (
                    logging.DEBUG
                    if request.url.path in {"/healthz", "/api/v1/health"}
                    else logging.WARNING
                    if code == 503
                    else logging.INFO
                )
                with logging_config.bind_task(getattr(request.state, "task_id", None)):
                    log.log(
                        level,
                        "request_completed",
                        extra={
                            "method": request.method,
                            "path": route,
                            "status_code": code,
                            "duration_ms": duration * 1000,
                        },
                    )
                logging_config.REQUEST_ID.reset(token)

    app.include_router(hello.router)
    app.include_router(health.router)
    app.include_router(version.router)
    app.include_router(process.router)
    app.include_router(tasks.router)
    return app


app = create_app()
