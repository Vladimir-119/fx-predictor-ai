"""JSON-контракт логов и связь HTTP-запроса с request_id."""

import json
import logging
from io import StringIO

from src.logging_config import JsonFormatter


async def test_request_log_matches_response_request_id(client, monkeypatch):
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
