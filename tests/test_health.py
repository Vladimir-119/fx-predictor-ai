"""Успешные и проблемные сценарии обязательных health-ручек."""

import asyncio
import json
import logging
from io import StringIO

from httpx import ASGITransport, AsyncClient

from src.logging_config import REQUEST_ID, JsonFormatter


async def test_liveness_returns_ok_without_database_query(client, pool):
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    pool.fetchval.assert_not_awaited()


async def test_health_report_healthy_with_postgres_version_and_timing(client, settings, pool):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"
    assert body["dependencies"]["application"]["version"] == settings.version
    postgres = body["dependencies"]["postgres"]
    assert postgres["status"] == "healthy"
    assert postgres["version"] == "16.10"
    assert postgres["response_time_ms"] >= 0
    pool.fetchval.assert_awaited_once_with("SELECT current_setting('server_version')")


async def test_health_report_degraded_when_postgres_unavailable(client, pool):
    pool.fetchval.side_effect = OSError("connection refused; secret-password")
    response = await client.get("/api/v1/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["postgres"]["status"] == "unavailable"
    assert body["dependencies"]["postgres"]["version"] is None
    assert body["dependencies"]["postgres"]["response_time_ms"] >= 0
    assert body["dependencies"]["application"]["status"] == "healthy"
    assert "secret-password" not in response.text
    assert (await client.get("/healthz")).status_code == 200


async def test_liveness_remains_responsive_during_database_check(client, pool):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_query(query):
        started.set()
        await release.wait()
        return "16.10"

    pool.fetchval.side_effect = slow_query
    task = asyncio.create_task(client.get("/api/v1/health"))
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        response = await asyncio.wait_for(client.get("/healthz"), timeout=0.1)
        assert response.status_code == 200
    finally:
        release.set()
        await task


async def test_unexpected_error_is_logged_and_context_is_restored(application, monkeypatch):
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
