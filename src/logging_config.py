"""JSON-логи в stdout с общим request_id для каждого HTTP-запроса."""

import contextvars
import json
import logging
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from opentelemetry import trace

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
TASK_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("task_id", default=None)
URL_CREDENTIALS = re.compile(r"([a-z][a-z0-9+.-]{0,31}://)[^/\s]+@", re.IGNORECASE)
NAMED_SECRETS = re.compile(
    r"(\b(?:password|passwd|secret|secret_key|aws_secret_access_key|api[_-]key|token|"
    r"authorization|x-amz-security-token|x-amz-signature)[\"']?\s*[:=]\s*)"
    r"(?:'[^']*'|\"[^\"]*\"|[^\s,;&]+)",
    re.IGNORECASE,
)
AUTH_TOKEN = re.compile(r"\b(Bearer|Basic)\s+[a-z0-9._~+/=-]+", re.IGNORECASE)


def redact_secrets(value: str) -> str:
    """Скрыть credentials в URL и явно названные секреты, сохранив traceback."""
    value = URL_CREDENTIALS.sub(r"\1***@", value)
    value = AUTH_TOKEN.sub(r"\1 ***", value)
    return NAMED_SECRETS.sub(r"\1***", value)


@contextmanager
def bind_task(task_id: object) -> Iterator[None]:
    """Связать все вложенные логи с задачей и восстановить прежний контекст."""
    token = TASK_ID.set(str(task_id) if task_id is not None else None)
    try:
        yield
    finally:
        TASK_ID.reset(token)


class JsonFormatter(logging.Formatter):
    """Сформировать одну JSON-запись на строку."""

    def format(self, record: logging.LogRecord) -> str:
        span = trace.get_current_span().get_span_context()
        data = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": REQUEST_ID.get(),
            "task_id": TASK_ID.get(),
            "service": os.getenv("OTEL_SERVICE_NAME", "fx-api"),
            "env": os.getenv("ENVIRONMENT", "development"),
            "trace_id": format(span.trace_id, "032x") if span.is_valid else None,
            "span_id": format(span.span_id, "016x") if span.is_valid else None,
        }
        for field in (
            "method",
            "path",
            "status_code",
            "duration_ms",
            "error_type",
            "app",
            "version",
            "model",
            "observations",
            "attempt",
            "attempts",
            "outcome",
            "result",
            "rejections",
            "published",
            "queue",
            "model_alias",
            "model_version",
            "queue_wait_seconds",
            "job_id",
            "count",
            "accepted",
            "rejected",
            "completed",
            "failed",
            "mode",
            "returned_trace_id",
        ):
            if hasattr(record, field):
                data[field] = getattr(record, field)
        if data["task_id"] is None and hasattr(record, "task_id"):
            data["task_id"] = str(record.task_id)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        for field, value in data.items():
            if isinstance(value, str):
                data[field] = redact_secrets(value)
        return json.dumps(data, ensure_ascii=False)


def setup_logging(level: str) -> None:
    """Настроить приложение и uvicorn в едином JSON-формате."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Библиотеки используют тот же stdout-handler, а не свой ConsoleFormatter.
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and name.startswith(
            (
                "faststream",
                "mlflow",
                "httpx",
                "httpcore",
                "aio_pika",
                "aiormq",
                "uvicorn",
                "opentelemetry",
            )
        ):
            logger.handlers = []
            logger.propagate = True
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers = []
    access_logger.propagate = False
