"""FastStream consumer и отдельный HTTP health-check в одном процессе."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID

import asyncpg
import pandas as pd
from aio_pika.exceptions import AMQPException
from fastapi import FastAPI
from faststream import AckPolicy
from faststream.rabbit import RabbitBroker, RabbitMessage
from opentelemetry import propagate, trace
from opentelemetry.trace import SpanKind, StatusCode
from pamqp.commands import Basic
from starlette.concurrency import run_in_threadpool

from src import db, logging_config, messaging, telemetry
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
        headers: dict[str, str] = {}
        propagate.inject(headers)
        confirmation = await asyncio.wait_for(
            self.broker.publish(
                {"message": body.hex() if isinstance(body, bytes) else body, "error": error},
                queue=messaging.dead_letter_queue(self.settings).name,
                persist=True,
                mandatory=True,
                headers=headers,
                correlation_id=logging_config.TASK_ID.get(),
                timeout=self.settings.publish_timeout_seconds,
            ),
            timeout=self.settings.publish_timeout_seconds,
        )
        if confirmation is False or isinstance(confirmation, Basic.Nack):
            raise messaging.QueueUnavailableError("DLQ did not accept the message")
        trace.get_current_span().add_event("task.dlq", {"reason": error})
        log.warning("task_sent_to_dlq", extra={"error_type": error})
        telemetry.metrics.worker_tasks.add(1, {"result": "dlq"})

    async def infer(self, task: TaskMessage, connection: asyncpg.Connection) -> ForecastResponse:
        """Пакетные истории не требуют свежести на сегодня; live-запросы требуют."""
        if task.task.history is None:
            return await prediction.forecast(
                task.task,
                self.model,
                connection,
                datetime.now(UTC),
                save=False,
                delay_seconds=self.settings.inference_delay_seconds,
            )
        history = pd.DataFrame([row.model_dump() for row in task.task.history])
        history["date"] = pd.to_datetime(history["date"])
        as_of = history.date.iloc[-1].date()
        target = next_publication_date(as_of)
        if task.task.forecast_date is not None and task.task.forecast_date != target:
            raise prediction.InvalidForecastDateError(f"Expected next publication: {target}")
        predicted = await prediction.infer_rate(
            self.model, history, self.settings.inference_delay_seconds
        )
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
        trace.get_current_span().set_attribute("task.attempt", attempt)
        log.info("task_attempt_started", extra={"attempt": attempt})
        try:
            result = await self.infer(task, connection)
        except Exception as exc:
            telemetry.record_error(trace.get_current_span(), exc)
            telemetry.metrics.worker_tasks.add(1, {"result": "error"})
            log.exception(
                "task_inference_failed",
                extra={"error_type": type(exc).__name__, "attempt": attempt},
            )
            if attempt >= self.settings.task_max_attempts:
                await self.fail(connection, task, body, type(exc).__name__)
                await message.ack()
            else:
                await asyncio.sleep(self.settings.retry_delay_seconds)
                await message.nack(requeue=True)
            return
        with telemetry.get_tracer().start_as_current_span(
            "db.task.complete", record_exception=False, set_status_on_exception=False
        ):
            await connection.execute(
                """UPDATE tasks SET status = 'COMPLETED', result = $2::jsonb,
                error = NULL, finished_at = now() WHERE task_id = $1""",
                task_id,
                result.model_dump_json(),
            )
        self.completed += 1
        telemetry.metrics.worker_tasks.add(1, {"result": "ok"})
        telemetry.metrics.task_duration.record(
            max(0, (datetime.now(UTC) - task.submitted_at).total_seconds())
        )
        log.info("task_completed", extra={"attempt": attempt})
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
        with telemetry.get_tracer().start_as_current_span(
            "db.task.fail", record_exception=False, set_status_on_exception=False
        ):
            await connection.execute(
                """UPDATE tasks SET status = 'FAILED', error = $2, finished_at = now()
                WHERE task_id = $1""",
                task.task.task_id,
                error,
            )

    async def handle(self, body: object, message: RabbitMessage) -> None:
        """Продолжить W3C-контекст исходной публикации, в том числе при redelivery."""
        headers = message.headers if isinstance(message.headers, dict) else {}
        correlation = getattr(message, "correlation_id", None)
        try:
            task_id = str(UUID(correlation)) if isinstance(correlation, str) else None
        except ValueError:
            task_id = None
        with (
            logging_config.bind_task(task_id),
            telemetry.get_tracer().start_as_current_span(
                "process",
                context=propagate.extract(headers),
                kind=SpanKind.CONSUMER,
                record_exception=False,
                set_status_on_exception=False,
            ) as span,
        ):
            span.set_attribute("messaging.destination.name", self.settings.tasks_queue)
            raw = getattr(message, "raw_message", message)
            redelivered = getattr(raw, "redelivered", False) is True
            span.set_attribute("messaging.message.redelivered", redelivered)
            if redelivered:
                telemetry.metrics.worker_redeliveries.add(1)
            published = headers.get("published_at_ns")
            if (
                not redelivered
                and isinstance(published, str)
                and published.isdecimal()
                and len(published) <= 20
            ):
                now = time.time_ns()
                # Не принимать произвольные timestamps из недоверенного сообщения.
                if now - 86_400_000_000_000 <= int(published) <= now:
                    with telemetry.get_tracer().start_as_current_span(
                        "queue.wait", start_time=int(published)
                    ):
                        pass
                    span.set_attribute("messaging.queue.wait_seconds", (now - int(published)) / 1e9)
            try:
                await self._handle(body, message)
            except Exception as exc:
                telemetry.record_error(span, exc)
                raise

    async def _handle(self, body: object, message: RabbitMessage) -> None:
        """Session lock исключает параллельный inference одного task_id на разных workers."""
        async with self.slots:
            self.active += 1
            telemetry.metrics.worker_active.add(1)
            try:
                try:
                    task = TaskMessage.model_validate(body)
                except ValueError:
                    span = trace.get_current_span()
                    span.record_exception(ValueError("Invalid task schema"))
                    span.set_status(StatusCode.ERROR)
                    await self.send_to_dlq(body, "Invalid task schema")
                    await message.ack()
                    return
                logging_config.TASK_ID.set(str(task.task.task_id))
                trace.get_current_span().set_attribute("task.id", str(task.task.task_id))
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
                span = trace.get_current_span()
                telemetry.record_error(span, exc)
                log.exception(
                    "task_dependency_unavailable", extra={"error_type": type(exc).__name__}
                )
                await asyncio.sleep(self.settings.retry_delay_seconds)
                await message.nack(requeue=True)
            finally:
                self.active -= 1
                telemetry.metrics.worker_active.add(-1)


def create_worker_app(settings: Settings | None = None) -> FastAPI:
    """Загрузить alias один раз, затем принимать сообщения до graceful shutdown."""
    settings = settings or get_settings()
    logging_config.setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging_config.setup_logging(settings.log_level)
        if not settings.model_alias:
            raise ValueError("Set MODEL_ALIAS=champion before starting inference-worker")
        runtime = telemetry.setup_telemetry(settings)
        pool = None
        broker = messaging.create_broker(settings)
        try:
            pool = await db.create_db_pool(settings)
            model = await run_in_threadpool(prediction.load_model, settings)
            log.info("model_loaded", extra={"model": model.status.model_dump()})
            worker = TaskWorker(settings, pool, model, broker)
            app.state.worker = worker
            broker.subscriber(
                messaging.task_queue(settings),
                ack_policy=AckPolicy.MANUAL,
                no_reply=True,
            )(worker.handle)
            await messaging.connect_broker(broker, settings)
            await broker.start()
            log.info("worker_started")
            yield
        finally:
            log.info("graceful_shutdown_started")
            await broker.stop()
            if pool is not None:
                await db.close_db_pool(pool)
            log.info("graceful_shutdown_completed")
            await asyncio.to_thread(runtime.shutdown)

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
