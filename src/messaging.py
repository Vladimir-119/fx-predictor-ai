"""Общая топология RabbitMQ и публикация с подтверждением брокера."""

import asyncio
import math

from aio_pika.exceptions import AMQPException
from faststream.message import StreamMessage, decode_message
from faststream.rabbit import Channel, RabbitBroker, RabbitQueue
from pamqp.commands import Basic

from src.config import Settings
from src.task_schemas import TaskMessage


class QueueUnavailableError(RuntimeError):
    """Брокер не подтвердил приём; источник решает, ждать ему или вернуть 503."""


def decode_task_message(message: StreamMessage) -> object:
    """Повреждённый JSON тоже доходит до handler и DLQ, а не зависает unacked."""
    try:
        return decode_message(message)
    except ValueError:
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
    try:
        confirmation = await asyncio.wait_for(
            broker.publish(
                message.model_dump(mode="json"),
                queue=settings.tasks_queue,
                persist=True,
                mandatory=True,
                message_id=str(message.task.task_id),
                timeout=settings.publish_timeout_seconds,
            ),
            timeout=settings.publish_timeout_seconds,
        )
    except (AMQPException, OSError, TimeoutError) as exc:
        raise QueueUnavailableError("RabbitMQ did not accept the task") from exc
    if confirmation is False or isinstance(confirmation, Basic.Nack):
        raise QueueUnavailableError("RabbitMQ rejected the task")


async def publish_with_backpressure(
    broker: RabbitBroker, settings: Settings, message: TaskMessage
) -> int:
    """Повторять одну задачу; не читать следующую запись до успешной отправки."""
    rejections = 0
    while True:
        try:
            await publish_task(broker, settings, message)
            return rejections
        except QueueUnavailableError:
            rejections += 1
            await asyncio.sleep(settings.retry_delay_seconds)


def retry_after(settings: Settings) -> str:
    """Подождать один настроенный цикл обработки; минимум одна секунда HTTP."""
    return str(max(1, math.ceil(settings.inference_delay_seconds)))
