"""MLflow + S3 + PostgreSQL + API в одноразовом стеке CI."""

import os
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

import asyncpg
import numpy as np
import pytest
from httpx import AsyncClient
from mlflow import MlflowClient

from src.schemas import RateObservation
from src.services.rates import expected_latest_date, is_publication_day, save_rates

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("MLOPS_TEST_DATABASE") != "1",
        reason="Requires a disposable database: MLOPS_TEST_DATABASE=1",
    ),
]


async def test_live_forecast_uses_registered_version_and_rate_import_is_idempotent():
    """Записать тестовые данные только в явно обозначенную одноразовую БД."""
    user = quote(os.environ["POSTGRES_USER"], safe="")
    password = quote(os.environ["POSTGRES_PASSWORD"], safe="")
    database = quote(os.environ["POSTGRES_DB"], safe="")
    port = int(os.getenv("POSTGRES_PORT", "5433"))
    connection_url = os.getenv(
        "TEST_DATABASE_URL", f"postgresql://{user}:{password}@127.0.0.1:{port}/{database}"
    )
    pool = await asyncpg.create_pool(connection_url, min_size=1, max_size=2)
    latest = expected_latest_date(datetime.now(UTC))
    days = [latest - timedelta(days=offset) for offset in range(100)]
    days = sorted(day for day in days if is_publication_day(day))
    observations = [
        RateObservation(
            base="EUR", quote="USD", rate_date=day, value=1.1 + 0.01 * float(np.sin(index / 8))
        )
        for index, day in enumerate(days)
    ]
    try:
        await save_rates(pool, observations)
        first_count = await pool.fetchval("SELECT count(*) FROM exchange_rates")
        await save_rates(pool, observations)
        assert await pool.fetchval("SELECT count(*) FROM exchange_rates") == first_count
        client = MlflowClient(tracking_uri=os.environ["MLFLOW_TEST_URI"])
        selected = client.get_model_version_by_alias("fx-predictor-ai-forecaster", "champion")
        async with AsyncClient(base_url=os.environ["API_BASE_URL"], trust_env=False) as http:
            status = await http.get("/api/v1/model")
            assert status.status_code == 200
            assert status.json()["version"] == selected.version
            response = await http.post("/process", json={})
            assert response.status_code == 200
            body = response.json()
            assert body["model"]["version"] == selected.version
            assert body["as_of"] == latest.isoformat()
            assert body["predicted_rate"] > 0
            assert (await http.get("/api/v1/rates")).json()["stale"] is False
        saved = await pool.fetchrow(
            "SELECT model_version, predicted_rate FROM predictions ORDER BY id DESC LIMIT 1"
        )
        assert saved["model_version"] == selected.version
        assert saved["predicted_rate"] == body["predicted_rate"]
    finally:
        await pool.close()
