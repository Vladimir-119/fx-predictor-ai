"""JSON-контракт логов и связь HTTP-запроса с request_id."""

import json
import logging
import sys
from io import StringIO

from httpx import ASGITransport, AsyncClient

from src.logging_config import REQUEST_ID, JsonFormatter, redact_secrets


def test_json_logs_hide_credentials_without_losing_exception_context():
    try:
        raise RuntimeError(
            "postgresql://user:db-secret@postgres:5432/app "
            "password='secret with spaces' Authorization: Bearer access-token "
            "Authorization: Basic dXNlcjpwYXNz X-Amz-Signature=signed-secret"
        )
    except RuntimeError:
        record = logging.LogRecord(
            "test", logging.ERROR, __file__, 1, "Connection failed", (), sys.exc_info()
        )
    result = JsonFormatter().format(record)
    parsed = json.loads(result)
    for secret in (
        "db-secret",
        "secret with spaces",
        "access-token",
        "dXNlcjpwYXNz",
        "signed-secret",
    ):
        assert secret not in result
    assert parsed["message"] == "Connection failed"
    assert "RuntimeError" in parsed["exception"]
    assert "postgresql://***@postgres:5432/app" in parsed["exception"]
    assert redact_secrets("a" * 10000) == "a" * 10000


async def test_request_logs_success_failure_and_restore_context(client, application, monkeypatch):
    access_logger = logging.getLogger("uvicorn.access")
    assert access_logger.propagate is False
    # pytest может добавить свои capture-handlers после настройки приложения.
    assert not any(
        isinstance(handler.formatter, JsonFormatter) for handler in access_logger.handlers
    )
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    monkeypatch.setattr(logging.getLogger(), "handlers", [handler])
    response = await client.get("/healthz")
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    completed = next(record for record in records if record["message"] == "request_completed")
    assert completed["request_id"] == response.headers["X-Request-ID"]
    assert completed["path"] == "/healthz"
    assert completed["status_code"] == 200
    assert completed["duration_ms"] >= 0

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    monkeypatch.setattr(logging.getLogger(), "handlers", [handler])

    @application.get("/test-error")
    async def fail():
        raise RuntimeError("test exception")

    async with AsyncClient(
        transport=ASGITransport(app=application, raise_app_exceptions=False),
        base_url="http://test",
    ) as http:
        response = await http.get("/test-error")
    assert response.status_code == 500
    assert REQUEST_ID.get() is None
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    failed = next(record for record in records if record["message"] == "request_failed")
    assert failed["status_code"] == 500
    assert failed["request_id"]
    assert "RuntimeError: test exception" in failed["exception"]
