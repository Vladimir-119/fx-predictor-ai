"""FastStream consumer и отдельный HTTP health-check в одном процессе."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import asyncpg
import pandas as pd
from aio_pika.exceptions import AMQPException
from fastapi import FastAPI
from faststream import AckPolicy
from faststream.rabbit import RabbitBroker, RabbitMessage
from pamqp.commands import Basic
from starlette.concurrency import run_in_threadpool

from src import db, logging_config, messaging
from src.config import Settings, get_settings
from src.schemas import ForecastResponse
from src.services import prediction
from src.services.rates import next_publication_date
from src.services.tasks import payload_hash
from src.task_schemas import TaskMessage

log = logging.getLogger(__name__)


class TaskWorker:
    """Ограничить вычисления; обработать повторную доставку по постоянному task_id."""

    def __init__(
        self,
        settings: Settings,
        pool: asyncpg.Pool,
        model: prediction.LoadedModel,
        broker: RabbitBroker,
    ) -> None:
        self.settings = settings
        self.pool = pool
        self.model = model
        self.broker = broker
        self.slots = asyncio.Semaphore(settings.worker_concurrency)
        self.active = 0
        self.completed = 0

    async def send_to_dlq(self, body: object, error: str) -> None:
        """Подтверждать исходное сообщение только после подтверждения DLQ."""
        confirmation = await asyncio.wait_for(
            self.broker.publish(
                {"message": body.hex() if isinstance(body, bytes) else body, "error": error},
                queue=messaging.dead_letter_queue(self.settings).name,
                persist=True,
                mandatory=True,
                timeout=self.settings.publish_timeout_seconds,
            ),
            timeout=self.settings.publish_timeout_seconds,
        )
        if confirmation is False or isinstance(confirmation, Basic.Nack):
            raise messaging.QueueUnavailableError("DLQ did not accept the message")

    async def infer(self, task: TaskMessage, connection: asyncpg.Connection) -> ForecastResponse:
        """Пакетные истории не требуют свежести на сегодня; live-запросы требуют."""
        await asyncio.sleep(self.settings.inference_delay_seconds)
        if task.task.history is None:
            return await prediction.forecast(
                task.task, self.model, connection, datetime.now(UTC), save=False
            )
        history = pd.DataFrame([row.model_dump() for row in task.task.history])
        history["date"] = pd.to_datetime(history["date"])
        as_of = history.date.iloc[-1].date()
        target = next_publication_date(as_of)
        if task.task.forecast_date is not None and task.task.forecast_date != target:
            raise prediction.InvalidForecastDateError(f"Expected next publication: {target}")
        predicted = await run_in_threadpool(prediction.predict_rate, self.model, history)
        return ForecastResponse(
            as_of=as_of,
            forecast_date=target,
            last_rate=float(history.rate.iloc[-1]),
            predicted_rate=predicted,
            model=self.model.status,
        )

    async def process(
        self,
        connection: asyncpg.Connection,
        task: TaskMessage,
        body: object,
        message: RabbitMessage,
    ) -> None:
        """Запись попытки коммитится до inference и переживает аварийную остановку."""
        task_id = task.task.task_id
        await connection.execute(
            """INSERT INTO tasks (task_id, payload_hash, submitted_at)
            VALUES ($1, $2, $3) ON CONFLICT (task_id) DO NOTHING""",
            task_id,
            payload_hash(task.task),
            task.submitted_at,
        )
        existing = await connection.fetchrow("SELECT * FROM tasks WHERE task_id = $1", task_id)
        if existing["payload_hash"] != payload_hash(task.task):
            await self.send_to_dlq(body, "task_id payload conflict")
            await message.ack()
            return
        if existing["status"] in {"COMPLETED", "FAILED"}:
            await message.ack()
            return
        if existing["attempts"] >= self.settings.task_max_attempts:
            await self.fail(connection, task, body, "Attempt limit reached")
            await message.ack()
            return
        attempt = await connection.fetchval(
            """UPDATE tasks SET attempts = attempts + 1, status = 'RUNNING',
            started_at = now() WHERE task_id = $1 RETURNING attempts""",
            task_id,
        )
        try:
            result = await self.infer(task, connection)
        except Exception as exc:  # noqa: BLE001 -- boundary: any model error must have bounded retries
            log.warning("task_inference_failed", extra={"error_type": type(exc).__name__})
            if attempt >= self.settings.task_max_attempts:
                await self.fail(connection, task, body, type(exc).__name__)
                await message.ack()
            else:
                await asyncio.sleep(self.settings.retry_delay_seconds)
                await message.nack(requeue=True)
            return
        await connection.execute(
            """UPDATE tasks SET status = 'COMPLETED', result = $2::jsonb,
            error = NULL, finished_at = now() WHERE task_id = $1""",
            task_id,
            result.model_dump_json(),
        )
        self.completed += 1
        await message.ack()

    async def fail(
        self,
        connection: asyncpg.Connection,
        task: TaskMessage,
        body: object,
        error: str,
    ) -> None:
        """Не ставить FAILED раньше, чем сообщение надёжно помещено в DLQ."""
        await self.send_to_dlq(body, error)
        await connection.execute(
            """UPDATE tasks SET status = 'FAILED', error = $2, finished_at = now()
            WHERE task_id = $1""",
            task.task.task_id,
            error,
        )

    async def handle(self, body: object, message: RabbitMessage) -> None:
        """Session lock исключает параллельный inference одного task_id на разных workers."""
        async with self.slots:
            self.active += 1
            try:
                try:
                    task = TaskMessage.model_validate(body)
                except ValueError:
                    await self.send_to_dlq(body, "Invalid task schema")
                    await message.ack()
                    return
                async with self.pool.acquire() as connection:
                    task_id = str(task.task.task_id)
                    locked = await connection.fetchval(
                        "SELECT pg_try_advisory_lock(hashtextextended($1, 0))", task_id
                    )
                    if not locked:
                        await asyncio.sleep(self.settings.retry_delay_seconds)
                        await message.nack(requeue=True)
                        return
                    try:
                        await self.process(connection, task, body, message)
                    finally:
                        await connection.execute(
                            "SELECT pg_advisory_unlock(hashtextextended($1, 0))", task_id
                        )
            except (
                asyncpg.PostgresError,
                AMQPException,
                OSError,
                TimeoutError,
                messaging.QueueUnavailableError,
            ) as exc:
                log.warning("task_dependency_unavailable", extra={"error_type": type(exc).__name__})
                await asyncio.sleep(self.settings.retry_delay_seconds)
                await message.nack(requeue=True)
            finally:
                self.active -= 1


def create_worker_app(settings: Settings | None = None) -> FastAPI:
    """Загрузить alias один раз, затем принимать сообщения до graceful shutdown."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging_config.setup_logging(settings.log_level)
        if not settings.model_alias:
            raise ValueError("Set MODEL_ALIAS=champion before starting inference-worker")
        pool = await db.create_db_pool(settings)
        broker = messaging.create_broker(settings)
        try:
            model = await run_in_threadpool(prediction.load_model, settings)
            worker = TaskWorker(settings, pool, model, broker)
            app.state.worker = worker
            broker.subscriber(
                messaging.task_queue(settings),
                ack_policy=AckPolicy.MANUAL,
                no_reply=True,
            )(worker.handle)
            await messaging.connect_broker(broker, settings)
            await broker.start()
            yield
        finally:
            await broker.stop()
            await db.close_db_pool(pool)

    app = FastAPI(title="fx-predictor-ai inference worker", lifespan=lifespan)

    @app.get("/healthz")
    async def health() -> dict[str, object]:
        worker = app.state.worker
        return {
            "status": "ok",
            "active": worker.active,
            "completed": worker.completed,
            "model": json.loads(worker.model.status.model_dump_json()),
        }

    return app


app = create_worker_app()
