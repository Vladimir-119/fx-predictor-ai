"""Контракты задач, идемпотентность, подтверждения и границы worker."""

import asyncio
import threading
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import asyncpg
import numpy as np
import pandas as pd
import pytest
from faststream import AckPolicy
from faststream.rabbit import TestRabbitBroker as RabbitTestBroker
from httpx import ASGITransport, AsyncClient
from pamqp.commands import Basic

from src import db, messaging
from src.inference_worker import TaskWorker, create_worker_app
from src.schemas import ModelStatus
from src.services import prediction, tasks
from src.task_schemas import TaskMessage, TaskRequest


@pytest.fixture
def message():
    history = [
        {"date": day.date(), "rate": 0.8} for day in pd.bdate_range(end="2026-10-07", periods=20)
    ]
    return TaskMessage(task=TaskRequest(history=history))


@pytest.fixture
def store(message):
    connection = AsyncMock(spec=asyncpg.Connection)
    connection.transaction = MagicMock()
    connection.fetchrow.return_value = {
        "payload_hash": tasks.payload_hash(message.task),
        "status": "QUEUED",
        "attempts": 0,
    }
    connection.fetchval.return_value = 1
    pool = MagicMock(spec=asyncpg.Pool)
    pool.acquire.return_value.__aenter__.return_value = connection
    return pool, connection


@pytest.fixture
def loaded():
    return prediction.LoadedModel.model_construct(
        predictor=SimpleNamespace(predict=lambda inputs: np.array([0.001])),
        status=ModelStatus(
            name="test-model", version="1", run_id="test", alias_at_startup="champion"
        ),
    )


def test_topology_confirms_prefetch_and_retry_after(settings):
    queue = messaging.task_queue(settings)
    assert queue.durable
    assert queue.arguments["x-max-length"] == 20
    assert queue.arguments["x-overflow"] == "reject-publish"
    settings.queue_limit = 0
    assert "x-max-length" not in messaging.task_queue(settings).arguments
    assert messaging.dead_letter_queue(settings).name.endswith(".dlq")
    settings.inference_delay_seconds = 1.5
    assert messaging.retry_after(settings) == "2"
    assert messaging.create_broker(settings)


async def test_publish_and_source_backpressure(settings, message, monkeypatch):
    broker = MagicMock()
    broker.publish = AsyncMock(return_value=Basic.Ack())
    await messaging.publish_task(broker, settings, message)
    assert broker.publish.call_args.kwargs["persist"] is True
    for result in (Basic.Nack(), False):
        broker.publish.return_value = result
        with pytest.raises(messaging.QueueUnavailableError):
            await messaging.publish_task(broker, settings, message)
    broker.publish.side_effect = TimeoutError()
    with pytest.raises(messaging.QueueUnavailableError):
        await messaging.publish_task(broker, settings, message)
    sender = AsyncMock(side_effect=[messaging.QueueUnavailableError(), None])
    sleeper = AsyncMock()
    monkeypatch.setattr(messaging, "publish_task", sender)
    monkeypatch.setattr(messaging.asyncio, "sleep", sleeper)
    assert await messaging.publish_with_backpressure(broker, settings, message) == 1
    assert sender.call_args_list[0].args[-1] is sender.call_args_list[1].args[-1]
    sleeper.assert_awaited_once_with(settings.retry_delay_seconds)
    connector = MagicMock(connect=AsyncMock(), declare_queue=AsyncMock())
    await messaging.connect_broker(connector, settings)
    assert connector.declare_queue.await_count == 2


async def test_admission_is_transactional_and_duplicate_ids_do_not_publish(
    settings, message, store, monkeypatch
):
    pool, connection = store
    sender = AsyncMock()
    monkeypatch.setattr(tasks, "publish_task", sender)
    await tasks.submit_task(pool, MagicMock(), settings, message.task)
    sender.assert_awaited_once()
    sender.side_effect = messaging.QueueUnavailableError()
    with pytest.raises(messaging.QueueUnavailableError):
        await tasks.submit_task(pool, MagicMock(), settings, message.task)
    exit_args = connection.transaction.return_value.__aexit__.call_args.args
    assert exit_args[0] is messaging.QueueUnavailableError
    connection.fetchval.return_value = None
    sender.reset_mock()
    await tasks.submit_task(pool, MagicMock(), settings, message.task)
    sender.assert_not_awaited()
    connection.fetchrow.return_value["payload_hash"] = "different"
    with pytest.raises(tasks.TaskConflictError):
        await tasks.submit_task(pool, MagicMock(), settings, message.task)


async def test_task_api_accepts_reports_overload_and_validates_ids(
    client, application, settings, message, monkeypatch
):
    task_id = str(message.task.task_id)
    response = await client.post("/api/v1/tasks", json={})
    assert response.status_code == 503 and response.headers["Retry-After"] == "1"
    application.state.task_broker = SimpleNamespace(stop=AsyncMock())
    submitter = AsyncMock(return_value={"task_id": task_id, "status": "QUEUED"})
    monkeypatch.setattr(tasks, "submit_task", submitter)
    response = await client.post("/api/v1/tasks", json=message.task.model_dump(mode="json"))
    assert response.status_code == 202 and response.json()["task_id"] == task_id
    for error, status in (
        (messaging.QueueUnavailableError(), 503),
        (tasks.TaskConflictError("id"), 409),
    ):
        submitter.side_effect = error
        assert (await client.post("/api/v1/tasks", json={})).status_code == status
    reader = AsyncMock(return_value=None)
    monkeypatch.setattr(tasks, "get_task", reader)
    assert (await client.get(f"/api/v1/tasks/{task_id}")).status_code == 404
    reader.side_effect = OSError()
    assert (await client.get(f"/api/v1/tasks/{task_id}")).status_code == 503
    reader.side_effect = None
    reader.return_value = {
        "task_id": task_id,
        "status": "QUEUED",
        "attempts": 0,
        "submitted_at": datetime.now(UTC),
    }
    assert (await client.get(f"/api/v1/tasks/{task_id}")).status_code == 200
    assert (await client.get("/api/v1/tasks/not-a-uuid")).status_code == 422
    assert (await client.post("/api/v1/tasks", json={"history": []})).status_code == 422


async def test_task_result_decodes_jsonb(store, message, loaded, settings):
    pool, connection = store
    pool.fetchrow.return_value = None
    assert await tasks.get_task(pool, str(uuid4())) is None
    settings.inference_delay_seconds = 0
    worker = TaskWorker(settings, pool, loaded, MagicMock())
    result = await worker.infer(message, connection)
    record = {
        "task_id": message.task.task_id,
        "status": "COMPLETED",
        "attempts": 1,
        "result": result.model_dump_json(),
        "submitted_at": datetime.now(UTC),
    }
    pool.fetchrow.return_value = record
    assert (await tasks.get_task(pool, str(message.task.task_id))).result.predicted_rate == 0.801
    record["result"] = None
    assert (await tasks.get_task(pool, str(message.task.task_id))).result is None


async def test_real_test_broker_runs_inference_and_acks_after_result(
    settings, store, loaded, message
):
    pool, connection = store
    settings.inference_delay_seconds = 0
    broker = messaging.create_broker(settings)
    worker = TaskWorker(settings, pool, loaded, broker)
    broker.subscriber(messaging.task_queue(settings), ack_policy=AckPolicy.MANUAL, no_reply=True)(
        worker.handle
    )
    dead_letters = []

    @broker.subscriber(messaging.dead_letter_queue(settings), no_reply=True)
    async def collect_dead_letter(body):
        dead_letters.append(body)

    async with RabbitTestBroker(broker):
        await broker.publish(message.model_dump(mode="json"), queue=settings.tasks_queue)
        await broker.publish(
            b"{broken JSON", queue=settings.tasks_queue, content_type="application/json"
        )
    assert worker.completed == 1 and worker.active == 0
    assert dead_letters == [{"message": b"{broken JSON".hex(), "error": "Invalid task schema"}]
    assert any("COMPLETED" in call.args[0] for call in connection.execute.call_args_list)


async def test_worker_deduplicates_locks_and_retries_before_dlq(settings, store, loaded, message):
    pool, connection = store
    settings.inference_delay_seconds = 0
    settings.retry_delay_seconds = 0.001
    broker = SimpleNamespace(publish=AsyncMock())
    worker = TaskWorker(settings, pool, loaded, broker)
    delivery = AsyncMock()
    body = message.model_dump(mode="json")
    await worker.handle(body, delivery)
    delivery.ack.assert_awaited_once()
    calls = [call.args[0] for call in connection.execute.call_args_list]
    assert "COMPLETED" in calls[-2] and "advisory_unlock" in calls[-1]
    for status in ("COMPLETED", "FAILED"):
        connection.fetchrow.return_value["status"] = status
        await worker.handle(body, delivery)
    assert worker.completed == 1
    connection.fetchrow.return_value["status"] = "QUEUED"
    connection.fetchrow.return_value["payload_hash"] = "other"
    await worker.handle(body, delivery)
    assert broker.publish.call_args.args[0]["error"] == "task_id payload conflict"
    connection.fetchrow.return_value["payload_hash"] = tasks.payload_hash(message.task)
    connection.fetchrow.return_value["attempts"] = settings.task_max_attempts
    await worker.handle(body, delivery)
    assert broker.publish.call_args.args[0]["error"] == "Attempt limit reached"
    connection.fetchrow.return_value["attempts"] = 0
    worker.infer = AsyncMock(side_effect=ValueError("model failed"))
    connection.fetchval.return_value = 1
    await worker.handle(body, delivery)
    delivery.nack.assert_awaited_with(requeue=True)
    connection.fetchval.side_effect = [True, settings.task_max_attempts]
    await worker.handle(body, delivery)
    assert broker.publish.call_args.args[0]["error"] == "ValueError"
    connection.fetchval.side_effect = None
    connection.fetchval.return_value = False
    await worker.handle(body, delivery)
    connection.fetchval.side_effect = asyncpg.PostgresError("offline")
    await worker.handle(body, delivery)
    assert worker.active == 0


async def test_invalid_messages_and_dlq_failure_are_not_silently_acked(settings, store, loaded):
    pool, _ = store
    settings.retry_delay_seconds = 0.001
    broker = SimpleNamespace(publish=AsyncMock(return_value=Basic.Nack()))
    worker = TaskWorker(settings, pool, loaded, broker)
    delivery = AsyncMock()
    await worker.handle({"bad": "message"}, delivery)
    delivery.ack.assert_not_awaited()
    delivery.nack.assert_awaited_once()
    broker.publish.return_value = Basic.Ack()
    await worker.handle({"bad": "message"}, delivery)
    delivery.ack.assert_awaited_once()
    for malformed in ("not JSON", b"\xff"):
        await worker.handle(malformed, delivery)
    assert delivery.ack.await_count == 3


async def test_worker_live_history_and_horizon(settings, store, loaded, message, monkeypatch):
    pool, connection = store
    settings.inference_delay_seconds = 0
    worker = TaskWorker(settings, pool, loaded, MagicMock())
    result = await worker.infer(message, connection)
    assert result.forecast_date == date(2026, 10, 8)
    message.task.forecast_date = date(2026, 10, 9)
    with pytest.raises(prediction.InvalidForecastDateError):
        await worker.infer(message, connection)
    message.task.history = None
    forecaster = AsyncMock(return_value=result)
    monkeypatch.setattr(prediction, "forecast", forecaster)
    await worker.infer(message, connection)
    assert forecaster.call_args.kwargs["save"] is False


async def test_worker_startup_loads_once_health_is_responsive_and_shutdown_drains(
    settings, store, loaded, message, monkeypatch
):
    pool, connection = store
    settings.model_alias = "champion"
    settings.inference_delay_seconds = 0
    started = threading.Event()
    release = threading.Event()

    def slow_predict(inputs):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("Test inference was not released")
        return np.array([0.001])

    loaded.predictor.predict = slow_predict
    app = create_worker_app(settings)
    loader = MagicMock(return_value=loaded)
    broker = messaging.create_broker(settings)
    monkeypatch.setattr(db, "create_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(prediction, "load_model", loader)
    monkeypatch.setattr(messaging, "create_broker", lambda config: broker)
    monkeypatch.setattr(messaging, "connect_broker", AsyncMock())
    monkeypatch.setattr(broker, "start", AsyncMock())
    monkeypatch.setattr(broker, "stop", AsyncMock())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            running = asyncio.create_task(app.state.worker.infer(message, connection))
            try:
                assert await asyncio.to_thread(started.wait, 3)
                response = await asyncio.wait_for(client.get("/healthz"), timeout=0.5)
                assert response.json()["model"]["version"] == "1"
                assert not running.done()
            finally:
                release.set()
                await running
    loader.assert_called_once()
    broker.stop.assert_awaited_once()
    pool.close.assert_awaited_once()
    settings.model_alias = ""
    with pytest.raises(ValueError, match="MODEL_ALIAS"):
        async with create_worker_app(settings).router.lifespan_context(app):
            pass
    settings.model_alias = "champion"
    loader.side_effect = ValueError("bad model")
    with pytest.raises(ValueError):
        async with create_worker_app(settings).router.lifespan_context(app):
            pass
    assert pool.close.await_count == 2


async def test_api_broker_lifecycle_success_and_failure(application, settings, monkeypatch):
    settings.tasks_enabled = True
    broker = SimpleNamespace(stop=AsyncMock())
    connector = AsyncMock()
    monkeypatch.setattr(messaging, "create_broker", lambda config: broker)
    monkeypatch.setattr(messaging, "connect_broker", connector)
    async with application.router.lifespan_context(application):
        assert application.state.task_broker is broker
    connector.side_effect = ConnectionError()
    async with application.router.lifespan_context(application):
        assert application.state.task_broker is None
    assert broker.stop.await_count == 2
