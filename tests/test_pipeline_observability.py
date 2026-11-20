"""Настоящие SDK-спаны всего пути без внешних хранилищ и брокера."""

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from pamqp.commands import Basic

from src import logging_config, messaging, telemetry
from src.app import create_app
from src.inference_worker import TaskWorker
from src.schemas import ModelStatus
from src.services.prediction import LoadedModel
from src.services.tasks import payload_hash
from src.task_schemas import TaskMessage, TaskRequest


@pytest.fixture
def captured_telemetry(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    spans = InMemorySpanExporter()
    traces = TracerProvider(sampler=ALWAYS_ON)
    traces.add_span_processor(SimpleSpanProcessor(spans))
    reader = InMemoryMetricReader()
    meters = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(telemetry, "_runtime", telemetry.Telemetry(traces, meters))
    monkeypatch.setattr(
        telemetry, "metrics", telemetry.create_instruments(meters.get_meter("test"))
    )
    yield spans, reader
    traces.shutdown()
    meters.shutdown()


def make_task():
    return TaskRequest(
        history=[
            {"date": day.date(), "rate": 0.8}
            for day in pd.bdate_range(end="2026-10-07", periods=20)
        ]
    )


def make_pool(task):
    connection = AsyncMock()
    connection.fetchval.return_value = task.task_id
    connection.fetchrow.return_value = {
        "payload_hash": payload_hash(task),
        "status": "QUEUED",
        "attempts": 0,
    }

    @asynccontextmanager
    async def transaction():
        yield

    connection.transaction = transaction
    pool = MagicMock()

    @asynccontextmanager
    async def acquire():
        yield connection

    pool.acquire = acquire
    return pool, connection


def loaded_model():
    def predict(_inputs):
        # Проверяем контекст именно внутри CPU-функции в другом потоке.
        span = trace.get_current_span()
        assert span.get_span_context().is_valid
        span.add_event("model.thread.context_preserved")
        logging.getLogger(__name__).info("model_thread_executed")
        return np.array([0.001])

    return LoadedModel.model_construct(
        predictor=SimpleNamespace(predict=predict),
        status=ModelStatus(name="test", version="7", run_id="test", alias_at_startup="champion"),
    )


async def test_api_publish_worker_inference_database_share_one_trace(settings, captured_telemetry):
    spans, reader = captured_telemetry
    task = make_task()
    pool, _ = make_pool(task)
    broker = SimpleNamespace(publish=AsyncMock(return_value=Basic.Ack()))
    app = create_app(settings)
    app.state.db_pool = pool
    app.state.task_broker = broker
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/api/v1/tasks", json=task.model_dump(mode="json"))
    assert response.status_code == 202
    published = broker.publish.call_args
    assert published.kwargs["correlation_id"] == str(task.task_id)
    assert "traceparent" in published.kwargs["headers"]

    worker_pool, connection = make_pool(task)
    connection.fetchval.side_effect = [True, 1]
    settings.inference_delay_seconds = 0
    worker = TaskWorker(settings, worker_pool, loaded_model(), broker)
    delivery = SimpleNamespace(
        headers=published.kwargs["headers"],
        correlation_id=str(task.task_id),
        redelivered=False,
        ack=AsyncMock(),
        nack=AsyncMock(),
    )
    await worker.handle(published.args[0], delivery)
    delivery.ack.assert_awaited_once()
    assert worker.completed == 1 and worker.active == 0
    finished = spans.get_finished_spans()
    by_name = {span.name: span for span in finished}
    assert {
        "POST /api/v1/tasks",
        "publish",
        "queue.wait",
        "process",
        "inference",
        "db.task.complete",
    } <= by_name.keys()
    assert len({span.context.trace_id for span in finished}) == 1
    assert response.headers["X-Trace-ID"] == f"{finished[0].context.trace_id:032x}"
    assert by_name["process"].parent.span_id == by_name["publish"].context.span_id
    assert by_name["inference"].parent.span_id == by_name["process"].context.span_id
    assert by_name["inference"].attributes["model.version"] == "7"
    assert any(
        event.name == "model.thread.context_preserved" for event in by_name["inference"].events
    )
    assert logging_config.TASK_ID.get() is None
    for resource in reader.get_metrics_data().resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                assert all("task_id" not in point.attributes for point in metric.data.data_points)


async def test_backpressure_retries_and_api_503_are_visible(settings, captured_telemetry):
    spans, reader = captured_telemetry
    task = make_task()
    settings.retry_delay_seconds = 0.001
    broker = SimpleNamespace(publish=AsyncMock(side_effect=[Basic.Nack(), Basic.Ack()]))
    with telemetry.get_tracer().start_as_current_span("source.task"):
        assert (
            await messaging.publish_with_backpressure(broker, settings, TaskMessage(task=task)) == 1
        )
    app = create_app(settings)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.post("/api/v1/tasks", json=task.model_dump(mode="json"))
        assert (await client.request("ARBITRARY_METHOD", "/unknown/uuid-value")).status_code == 404
    assert response.status_code == 503
    finished = spans.get_finished_spans()
    assert any(span.name == "source.wait" for span in finished)
    assert any(span.name == "publish" and span.status.is_ok is False for span in finished)
    api_span = next(span for span in finished if span.name == "POST /api/v1/tasks")
    assert api_span.status.status_code == trace.StatusCode.ERROR
    assert any(event.name == "exception" for event in api_span.events)
    points = [
        point
        for resource in reader.get_metrics_data().resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == "fx_api_requests"
        for point in metric.data.data_points
    ]
    assert any(
        point.attributes["method"] == "OTHER" and point.attributes["route"] == "unmatched"
        for point in points
    )


async def test_redelivery_attempts_and_dlq_continue_the_original_trace(
    settings, captured_telemetry
):
    spans, reader = captured_telemetry
    task = make_task()
    task.forecast_date = task.history[-1].date
    envelope = TaskMessage(task=task)
    broker = SimpleNamespace(publish=AsyncMock(return_value=Basic.Ack()))
    await messaging.publish_task(broker, settings, envelope)
    published = broker.publish.call_args
    pool, connection = make_pool(task)
    connection.fetchval.side_effect = [True, 1, True, 2, True, 3]
    settings.retry_delay_seconds = 0.001
    worker = TaskWorker(settings, pool, loaded_model(), broker)
    delivery = SimpleNamespace(
        headers=published.kwargs["headers"],
        correlation_id=str(task.task_id),
        raw_message=SimpleNamespace(redelivered=False),
        ack=AsyncMock(),
        nack=AsyncMock(),
    )
    for attempt in range(3):
        delivery.raw_message.redelivered = attempt > 0
        await worker.handle(published.args[0], delivery)
    assert delivery.nack.await_count == 2
    delivery.ack.assert_awaited_once()
    assert broker.publish.call_args.kwargs["queue"].endswith(".dlq")
    finished = spans.get_finished_spans()
    process = [span for span in finished if span.name == "process"]
    assert [span.attributes["task.attempt"] for span in process] == [1, 2, 3]
    assert all(span.status.status_code == trace.StatusCode.ERROR for span in process)
    assert len({span.context.trace_id for span in finished}) == 1
    assert sum(span.name == "queue.wait" for span in finished) == 1
    assert any(event.name == "task.dlq" for event in process[-1].events)
    points = {
        metric.name: metric.data.data_points
        for resource in reader.get_metrics_data().resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert points["fx_worker_redeliveries"][0].value == 2
    assert logging_config.TASK_ID.get() is None
