"""Транзакционный приём задач и результаты с уникальным task_id."""

import hashlib
import json
from uuid import UUID

import asyncpg
from faststream.rabbit import RabbitBroker

from src.config import Settings
from src.messaging import publish_task
from src.task_schemas import TaskAccepted, TaskMessage, TaskRequest, TaskResult


class TaskConflictError(ValueError):
    """Один task_id нельзя использовать для разных входных данных."""


def payload_hash(task: TaskRequest) -> str:
    """Сравнивать вход без зависимого от повторной отправки timestamp."""
    content = json.dumps(task.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


async def submit_task(
    pool: asyncpg.Pool, broker: RabbitBroker, settings: Settings, task: TaskRequest
) -> TaskAccepted:
    """При отказе брокера INSERT откатывается вместе с транзакцией."""
    message = TaskMessage(task=task)
    async with pool.acquire() as connection:
        async with connection.transaction():
            inserted = await connection.fetchval(
                """INSERT INTO tasks (task_id, payload_hash, submitted_at)
                VALUES ($1, $2, $3) ON CONFLICT (task_id) DO NOTHING RETURNING task_id""",
                task.task_id,
                payload_hash(task),
                message.submitted_at,
            )
            if inserted is None:
                existing = await connection.fetchrow(
                    "SELECT payload_hash, status FROM tasks WHERE task_id = $1", task.task_id
                )
                if existing["payload_hash"] != payload_hash(task):
                    raise TaskConflictError("task_id already belongs to another request")
                return TaskAccepted(task_id=task.task_id, status=existing["status"])
            await publish_task(broker, settings, message)
    return TaskAccepted(task_id=task.task_id)


async def get_task(pool: asyncpg.Pool, task_id: str) -> TaskResult | None:
    """JSONB декодируется явно: asyncpg по умолчанию возвращает строку."""
    row = await pool.fetchrow("SELECT * FROM tasks WHERE task_id = $1", UUID(task_id))
    if row is None:
        return None
    data = dict(row)
    if data["result"] is not None:
        data["result"] = json.loads(data["result"])
    return TaskResult.model_validate(data)
