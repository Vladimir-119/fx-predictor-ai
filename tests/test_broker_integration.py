"""Настоящие RabbitMQ и PostgreSQL: переполнение, повторы, DLQ и drain."""

import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import aio_pika
import asyncpg
import numpy as np
import pandas as pd
import pytest
from faststream import AckPolicy
from faststream.rabbit import RabbitMessage
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from testcontainers.community.rabbitmq import RabbitMqContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from src import messaging
from src.app import create_app
from src.inference_worker import TaskWorker
from src.schemas import ModelStatus
from src.services.prediction import LoadedModel
from src.task_schemas import TaskMessage, TaskRequest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_BROKER_TESTS") != "1", reason="Run make broker-test"),
]


@pytest.fixture
def rabbit_url():
    supplied = os.getenv("BROKER_TEST_URL")
    if supplied:
        yield supplied
        return
    with RabbitMqContainer(
        "rabbitmq:4.1-management", username="test", password="test-password"
    ).with_env("RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS", "+S 2:2") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5672)
        yield f"amqp://test:test-password@{host}:{port}/"


@pytest.fixture
async def postgres_pool():
    with (
        DockerContainer("postgres:16-alpine")
        .with_env("POSTGRES_PASSWORD", "test-password")
        .with_exposed_ports(5432)
        .waiting_for(
            LogMessageWaitStrategy(
                "database system is ready to accept connections"
            ).with_startup_timeout(60)
        ) as container
    ):
        for _ in range(300):
            try:
                url = (
                    f"postgresql://postgres:test-password@{container.get_container_host_ip()}:"
                    f"{container.get_exposed_port(5432)}/postgres"
                )
                pool = await asyncpg.create_pool(url, min_size=1, max_size=5)
                break
            except (OSError, asyncpg.PostgresError):
                await asyncio.sleep(0.1)
        else:
            raise TimeoutError("PostgreSQL container did not become ready")
        try:
            await pool.execute(Path("sql/exchange_rates.sql").read_text(encoding="utf-8"))
            yield pool
        finally:
            await pool.close()


def task_message():
    history = [
        {"date": day.date(), "rate": 0.8} for day in pd.bdate_range(end="2026-10-07", periods=20)
    ]
    return TaskMessage(task=TaskRequest(history=history))


async def test_real_queue_overflow_prefetch_and_producer_resumes(settings, rabbit_url):
    settings.rabbitmq_url = SecretStr(rabbit_url)
    settings.tasks_queue = f"test-overflow-{uuid4().hex}"
    settings.queue_limit = 2
    settings.retry_delay_seconds = 0.02
    broker = messaging.create_broker(settings)
    await messaging.connect_broker(broker, settings)
    raw = await aio_pika.connect_robust(rabbit_url)
    channel = await raw.channel()
    try:
        await messaging.publish_task(broker, settings, task_message())
        await messaging.publish_task(broker, settings, task_message())
        with pytest.raises(messaging.QueueUnavailableError):
            await messaging.publish_task(broker, settings, task_message())
        queue = await channel.declare_queue(settings.tasks_queue, passive=True)
        assert queue.declaration_result.message_count == 2
        pending = asyncio.create_task(
            messaging.publish_with_backpressure(broker, settings, task_message())
        )
        await asyncio.sleep(0.1)
        assert not pending.done()
        delivery = await queue.get(timeout=2)
        await delivery.ack()
        assert await asyncio.wait_for(pending, 3) >= 1
        await channel.set_qos(prefetch_count=1)
        received = []
        arrived = asyncio.Event()

        async def consume(message):
            received.append(message)
            arrived.set()

        await queue.consume(consume)
        await asyncio.wait_for(arrived.wait(), 2)
        await asyncio.sleep(0.1)
        assert len(received) == 1
        await channel.close()
        channel = await raw.channel()
        queue = await channel.declare_queue(settings.tasks_queue, passive=True)
        assert queue.declaration_result.message_count == 2
        started = asyncio.Event()
        release = asyncio.Event()
        seen = []

        @broker.subscriber(
            messaging.task_queue(settings), ack_policy=AckPolicy.MANUAL, no_reply=True
        )
        async def held_handler(body, message: RabbitMessage):
            seen.append(body)
            started.set()
            await release.wait()
            await message.ack()

        await broker.start()
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.sleep(0.05)
        assert len(seen) == 1
        draining = asyncio.create_task(broker.stop())
        await asyncio.sleep(0.05)
        assert not draining.done()
        release.set()
        await asyncio.wait_for(draining, 3)
    finally:
        cleanup = await raw.channel()
        await cleanup.queue_delete(settings.tasks_queue)
        await cleanup.queue_delete(f"{settings.tasks_queue}.dlq")
        await raw.close()
        await broker.stop()


async def test_real_api_rollback_worker_idempotency_dlq_and_graceful_shutdown(
    settings, rabbit_url, postgres_pool
):
    settings.rabbitmq_url = SecretStr(rabbit_url)
    settings.tasks_queue = f"test-tasks-{uuid4().hex}"
    settings.queue_limit = 2
    settings.inference_delay_seconds = 0.05
    settings.retry_delay_seconds = 0.01
    model = LoadedModel.model_construct(
        predictor=SimpleNamespace(predict=lambda inputs: np.array([0.001])),
        status=ModelStatus(
            name="test", version="1", run_id="integration", alias_at_startup="champion"
        ),
    )
    broker = messaging.create_broker(settings)
    worker = TaskWorker(settings, postgres_pool, model, broker)
    broker.subscriber(messaging.task_queue(settings), ack_policy=AckPolicy.MANUAL, no_reply=True)(
        worker.handle
    )
    await messaging.connect_broker(broker, settings)
    app = create_app(settings)
    app.state.db_pool = postgres_pool
    app.state.task_broker = broker
    rejected = task_message()
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    try:
        for _ in range(2):
            await messaging.publish_task(broker, settings, task_message())
        started = time.perf_counter()
        response = await client.post("/api/v1/tasks", json=rejected.task.model_dump(mode="json"))
        assert response.status_code == 503 and response.headers["Retry-After"] == "1"
        assert time.perf_counter() - started < 2
        assert (
            await postgres_pool.fetchval(
                "SELECT count(*) FROM tasks WHERE task_id = $1", rejected.task.task_id
            )
            == 0
        )
        await broker.start()
        for _ in range(100):
            if worker.completed == 2:
                break
            await asyncio.sleep(0.05)
        assert worker.completed == 2
        body = rejected.task.model_dump(mode="json")
        assert (await client.post("/api/v1/tasks", json=body)).status_code == 202
        for _ in range(100):
            state = (await client.get(f"/api/v1/tasks/{rejected.task.task_id}")).json()
            if state["status"] == "COMPLETED":
                break
            await asyncio.sleep(0.05)
        assert state["result"]["model"]["version"] == "1"
        assert state["attempts"] == 1
        assert (await client.post("/api/v1/tasks", json=body)).json()["status"] == "COMPLETED"
        await messaging.publish_task(broker, settings, rejected)
        await asyncio.sleep(0.1)
        assert worker.completed == 3
        poison = task_message()
        poison.task.history.reverse()
        await messaging.publish_task(broker, settings, poison)
        for _ in range(100):
            row = await postgres_pool.fetchrow(
                "SELECT * FROM tasks WHERE task_id = $1", poison.task.task_id
            )
            if row and row["status"] == "FAILED":
                break
            await asyncio.sleep(0.05)
        assert row["status"] == "FAILED" and row["attempts"] == 3
        inspect = await aio_pika.connect_robust(rabbit_url)
        inspect_channel = await inspect.channel()
        dlq = await inspect_channel.declare_queue(f"{settings.tasks_queue}.dlq", passive=True)
        assert dlq.declaration_result.message_count == 1
        await inspect.close()
        settings.inference_delay_seconds = 0.2
        final = task_message()
        await messaging.publish_task(broker, settings, final)
        for _ in range(100):
            if worker.active:
                break
            await asyncio.sleep(0.01)
        assert worker.active == 1
        await broker.stop()
        assert (
            await postgres_pool.fetchval(
                "SELECT status FROM tasks WHERE task_id = $1", final.task.task_id
            )
            == "COMPLETED"
        )
    finally:
        await client.aclose()
        await broker.stop()
        raw = await aio_pika.connect_robust(rabbit_url)
        channel = await raw.channel()
        await channel.queue_delete(settings.tasks_queue)
        await channel.queue_delete(f"{settings.tasks_queue}.dlq")
        await raw.close()
