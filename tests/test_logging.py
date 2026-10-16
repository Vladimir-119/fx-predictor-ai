"""JSON-контракт логов и связь HTTP-запроса с request_id."""

import json
import logging
from io import StringIO

from httpx import ASGITransport, AsyncClient

from src.logging_config import REQUEST_ID, JsonFormatter


async def test_request_logs_success_failure_and_restore_context(client, application, monkeypatch):
    assert not logging.getLogger("uvicorn.access").handlers
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
