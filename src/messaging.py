"""Общая топология RabbitMQ и публикация с подтверждением брокера."""

import asyncio
import logging
import math
import time

from aio_pika.exceptions import AMQPException
from faststream.message import StreamMessage, decode_message
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue
from opentelemetry import propagate, trace
from opentelemetry.trace import SpanKind
from pamqp.commands import Basic

from src import logging_config, telemetry
from src.config import Settings
from src.task_schemas import TaskMessage

log = logging.getLogger(__name__)
MAX_TASK_MESSAGE_BYTES = 128 * 1024


class QueueUnavailableError(RuntimeError):
    """Брокер не подтвердил приём; источник решает, ждать ему или вернуть 503."""


def decode_task_message(message: StreamMessage) -> object:
    """Повреждённый JSON тоже доходит до handler и DLQ, а не зависает unacked."""
    if len(message.body) > MAX_TASK_MESSAGE_BYTES:
        # Не копировать большой вредоносный payload в JSON/hex и затем в DLQ.
        return {"rejected_message": "Task message is too large", "size_bytes": len(message.body)}
    try:
        return decode_message(message)
    except (ValueError, RecursionError):
        return message.body


def task_queue(settings: Settings) -> RabbitQueue:
    """Ноль отключает ограничение только для контрольного эксперимента."""
    arguments: dict[str, str | int] = {"x-queue-type": "classic"}
    if settings.queue_limit:
        arguments.update({"x-max-length": settings.queue_limit, "x-overflow": "reject-publish"})
    return RabbitQueue(settings.tasks_queue, durable=True, arguments=arguments)


def dead_letter_queue(settings: Settings) -> RabbitQueue:
    """Отдельная долговечная очередь для необрабатываемых сообщений."""
    return RabbitQueue(f"{settings.tasks_queue}.dlq", durable=True)


def create_broker(settings: Settings) -> RabbitBroker:
    """Prefetch ограничивает доставки, Semaphore worker ограничивает вычисления."""
    return RabbitBroker(
        settings.rabbitmq_url.get_secret_value(),
        decoder=decode_task_message,
        logger=logging.getLogger("faststream"),
        default_channel=Channel(
            prefetch_count=settings.prefetch_count,
            publisher_confirms=True,
            on_return_raises=True,
        ),
        graceful_timeout=settings.shutdown_timeout_seconds,
        timeout=settings.publish_timeout_seconds,
    )


async def connect_broker(broker: RabbitBroker, settings: Settings) -> None:
    """Объявить обе очереди до первой публикации."""
    await broker.connect()
    await broker.declare_queue(task_queue(settings))
    await broker.declare_queue(dead_letter_queue(settings))


async def publish_task(broker: RabbitBroker, settings: Settings, message: TaskMessage) -> None:
    """Ждать только подтверждения публикации, а не результата inference."""
    with (
        logging_config.bind_task(message.task.task_id),
        telemetry.get_tracer().start_as_current_span(
            "publish", kind=SpanKind.PRODUCER, record_exception=False, set_status_on_exception=False
        ) as span,
    ):
        span.set_attribute("messaging.destination.name", settings.tasks_queue)
        span.set_attribute("messaging.message.id", str(message.task.task_id))
        headers: dict[str, str] = {"published_at_ns": str(time.time_ns())}
        propagate.inject(headers)
        try:
            confirmation = await asyncio.wait_for(
                broker.publish(
                    message.model_dump(mode="json"),
                    queue=settings.tasks_queue,
                    persist=True,
                    mandatory=True,
                    message_id=str(message.task.task_id),
                    correlation_id=str(message.task.task_id),
                    headers=headers,
                    timeout=settings.publish_timeout_seconds,
                ),
                timeout=settings.publish_timeout_seconds,
            )
            if confirmation is False or isinstance(confirmation, Basic.Nack):
                raise QueueUnavailableError("RabbitMQ rejected the task")
        except (AMQPException, OSError, TimeoutError, QueueUnavailableError) as exc:
            # Текст исключения драйвера может содержать DSN: в трейс идёт только тип.
            telemetry.record_error(span, exc)
            log.warning("broker_publish_rejected", extra={"error_type": type(exc).__name__})
            raise QueueUnavailableError("RabbitMQ did not accept the task") from exc
        log.info("task_published")


async def publish_with_backpressure(
    broker: RabbitBroker, settings: Settings, message: TaskMessage
) -> int:
    """Повторять одну задачу; не читать следующую запись до успешной отправки."""
    rejections = 0
    while True:
        try:
            await publish_task(broker, settings, message)
            telemetry.metrics.source_published.add(1)
            return rejections
        except QueueUnavailableError:
            rejections += 1
            telemetry.metrics.source_rejections.add(1)
            trace.get_current_span().add_event("broker.rejected", {"rejections": rejections})
            with telemetry.get_tracer().start_as_current_span("source.wait"):
                started = time.perf_counter()
                # Первое ожидание и редкие обновления, без сотен одинаковых строк.
                if rejections == 1 or rejections % 20 == 0:
                    log.warning("source_backpressure", extra={"rejections": rejections})
                await asyncio.sleep(settings.retry_delay_seconds)
                telemetry.metrics.source_wait.add(time.perf_counter() - started)


def retry_after(settings: Settings) -> str:
    """Подождать один настроенный цикл обработки; минимум одна секунда HTTP."""
    return str(max(1, math.ceil(settings.inference_delay_seconds)))
