"""Проверки работающего контейнера с настоящим PostgreSQL.
API_BASE_URL включает эти тесты. EXPECT_POSTGRES задаёт состояние БД."""

import os
from importlib.metadata import version

import pytest
from httpx import AsyncClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("API_BASE_URL"), reason="API_BASE_URL is not set"),
]


@pytest.fixture
async def live_client():
    async with AsyncClient(
        base_url=os.environ["API_BASE_URL"], timeout=35, trust_env=False
    ) as client:
        yield client


async def test_live_liveness_and_version(live_client):
    liveness = await live_client.get("/healthz")
    assert liveness.status_code == 200
    assert liveness.json() == {"status": "ok"}
    response = await live_client.get("/api/v1/version")
    assert response.status_code == 200
    assert response.json() == {"app": "fx-predictor-ai", "version": version("fx-predictor-ai")}
    assert len(response.headers["X-Request-ID"]) == 32


async def test_live_postgres_status_version_and_timing(live_client):
    expected = os.getenv("EXPECT_POSTGRES", "healthy")
    assert expected in {"healthy", "unavailable"}
    response = await live_client.get("/api/v1/health")
    body = response.json()
    postgres = body["dependencies"]["postgres"]
    assert postgres["status"] == expected
    assert postgres["response_time_ms"] >= 0
    if expected == "healthy":
        assert response.status_code == 200
        assert body["status"] == "ok"
        assert postgres["version"]
        assert postgres["version"][0].isdigit()
    else:
        assert response.status_code == 503
        assert body["status"] == "degraded"
        assert postgres["version"] is None
        assert postgres["detail"] == "PostgreSQL is unavailable or the check timed out"
    assert body["dependencies"]["application"]["status"] == "healthy"
    assert body["dependencies"]["application"]["version"] == version("fx-predictor-ai")
