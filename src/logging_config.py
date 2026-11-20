"""JSON-логи в stdout с общим request_id для каждого HTTP-запроса."""

import contextvars
import json
import logging
import re
import sys
from datetime import UTC, datetime

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
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


class JsonFormatter(logging.Formatter):
    """Сформировать одну JSON-запись на строку."""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": REQUEST_ID.get(),
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
        ):
            if hasattr(record, field):
                data[field] = getattr(record, field)
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
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers = []
    access_logger.propagate = False
