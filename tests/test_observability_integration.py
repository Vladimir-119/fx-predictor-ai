"""Живой стенд: задача видна в API, Loki, Tempo, Prometheus и готовой Grafana."""

import asyncio
import json
import os
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from httpx import AsyncClient, HTTPError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_OBSERVABILITY_TESTS") != "1",
        reason="Start the observability profile and set RUN_OBSERVABILITY_TESTS=1",
    ),
]


async def wait_for_json(client, path, condition, deadline, description, params=None):
    """У всех ожиданий общий бюджет: не растягивать упавший CI на несколько минут."""
    last = "no response"
    while time.monotonic() < deadline:
        try:
            response = await client.get(
                path,
                params=params,
                timeout=min(5, max(0.1, deadline - time.monotonic())),
            )
            if response.status_code in {401, 403}:
                response.raise_for_status()
            last = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code == 200:
                data = response.json()
                if condition(data):
                    return data
        except HTTPError as exc:
            if getattr(exc, "response", None) is not None and exc.response.status_code in {
                401,
                403,
            }:
                raise
            last = type(exc).__name__
        await asyncio.sleep(0.5)
    pytest.fail(f"{description} was not observed within the 60-second budget; last: {last}")


def span_names(value):
    """Tempo может вернуть batches или resourceSpans: структура OTLP несущественна."""
    names = set()
    if isinstance(value, dict):
        if isinstance(value.get("name"), str):
            names.add(value["name"])
        for item in value.values():
            names.update(span_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(span_names(item))
    return names


async def test_live_task_correlates_all_signals_and_grafana_is_provisioned():
    api_url = os.getenv("API_BASE_URL", f"http://127.0.0.1:{os.getenv('APP_PORT', '8000')}")
    grafana_url = os.getenv("GRAFANA_URL", f"http://127.0.0.1:{os.getenv('GRAFANA_PORT', '3000')}")
    prometheus_url = os.getenv(
        "PROMETHEUS_URL", f"http://127.0.0.1:{os.getenv('PROMETHEUS_PORT', '9090')}"
    )
    credentials = (
        os.getenv("GRAFANA_USER", "admin"),
        os.getenv("GRAFANA_PASSWORD", "local-grafana-change-me"),
    )
    task_id = str(uuid4())
    days = []
    day = datetime.now(UTC).date() - timedelta(days=1)
    while len(days) < 20:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    history = [
        {"date": day.isoformat(), "rate": 0.85 + index * 0.0001}
        for index, day in enumerate(reversed(days))
    ]
    started_ns = time.time_ns() - 5_000_000_000
    deadline = time.monotonic() + 60
    async with (
        AsyncClient(base_url=api_url, timeout=5, trust_env=False) as api,
        AsyncClient(
            base_url=grafana_url,
            auth=credentials,
            headers={"Accept": "application/json"},
            timeout=5,
            trust_env=False,
        ) as grafana,
        AsyncClient(base_url=prometheus_url, timeout=5, trust_env=False) as prometheus,
    ):
        health = await grafana.get("/api/health")
        health.raise_for_status()
        assert health.json()["database"] == "ok"
        datasources = await grafana.get("/api/datasources")
        datasources.raise_for_status()
        assert {"prometheus", "loki", "tempo"} <= {source["uid"] for source in datasources.json()}
        rules = await grafana.get("/api/v1/provisioning/alert-rules")
        rules.raise_for_status()
        assert {"fx-queue-near-capacity", "fx-dlq-not-empty"} <= {
            rule["uid"] for rule in rules.json()
        }
        dashboard = await grafana.get("/api/dashboards/uid/fx-pipeline")
        dashboard.raise_for_status()
        assert dashboard.json()["dashboard"]["panels"]

        submitted = await api.post("/api/v1/tasks", json={"task_id": task_id, "history": history})
        assert submitted.status_code == 202, submitted.text
        trace_id = submitted.headers.get("X-Trace-ID", "")
        assert len(trace_id) == 32 and int(trace_id, 16) > 0
        result = await wait_for_json(
            api,
            f"/api/v1/tasks/{task_id}",
            lambda data: data.get("status") in {"COMPLETED", "FAILED"},
            deadline,
            "completed inference task",
        )
        assert result["status"] == "COMPLETED", result
        assert result["attempts"] == 1
        assert result["result"]["predicted_rate"] > 0
        assert result["result"]["model"]["version"]

        trace = await wait_for_json(
            grafana,
            f"/api/datasources/proxy/uid/tempo/api/traces/{trace_id}",
            lambda data: (
                {"process", "inference", "db.task.complete"} <= span_names(data)
                and "fx-api" in json.dumps(data)
                and "fx-inference-worker" in json.dumps(data)
            ),
            deadline,
            "distributed trace through API, worker, inference and database",
        )
        trace_text = json.dumps(trace)
        assert "fx-api" in trace_text
        assert "fx-inference-worker" in trace_text
        log_query = '{service=~"fx-api|fx-inference-worker"} | json | task_id=' + json.dumps(
            task_id
        )

        def both_service_logs(data):
            streams = data.get("data", {}).get("result", [])
            services = {stream["stream"].get("service") for stream in streams}
            records = [json.loads(line) for stream in streams for _, line in stream["values"]]
            return {"fx-api", "fx-inference-worker"} <= services and any(
                record.get("message") == "task_completed" and record.get("trace_id") == trace_id
                for record in records
            )

        logs = await wait_for_json(
            grafana,
            "/api/datasources/proxy/uid/loki/loki/api/v1/query_range",
            both_service_logs,
            deadline,
            "task-correlated API and worker logs in Loki",
            params={
                "query": log_query,
                "start": str(started_ns),
                "end": str(started_ns + 70_000_000_000),
                "limit": "100",
            },
        )
        records = [
            json.loads(line) for stream in logs["data"]["result"] for _, line in stream["values"]
        ]
        assert all(record["task_id"] == task_id for record in records)
        assert any(
            record["message"] == "task_completed" and record["trace_id"] == trace_id
            for record in records
        )
        assert any(
            record["message"] == "request_completed" and record["trace_id"] == trace_id
            for record in records
        )

        for query in (
            'sum(fx_api_requests_total{route="/api/v1/tasks",status_code="202"})',
            "rabbitmq_detailed_queue_messages_ready{queue="
            + json.dumps(os.getenv("TASKS_QUEUE", "fx-predictor-ai.tasks"))
            + "}",
        ):
            metric = await wait_for_json(
                prometheus,
                "/api/v1/query",
                lambda data: bool(data.get("data", {}).get("result")),
                deadline,
                f"Prometheus metric {query}",
                params={"query": query},
            )
            assert metric["status"] == "success"
            assert float(metric["data"]["result"][0]["value"][1]) >= 0
