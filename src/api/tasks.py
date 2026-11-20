"""API приёма задач: перегруженный брокер означает быстрый отказ."""

from uuid import UUID

from asyncpg import PostgresError
from fastapi import APIRouter, HTTPException, Request

from src import logging_config
from src.messaging import QueueUnavailableError, retry_after
from src.services import tasks
from src.task_schemas import TaskAccepted, TaskRequest, TaskResult

router = APIRouter(prefix="/api/v1/tasks", tags=["Tasks"])


@router.post("", response_model=TaskAccepted, status_code=202)
async def submit(body: TaskRequest, request: Request) -> TaskAccepted:
    """Принять задачу; при перегрузке не ждать свободного места в очереди."""
    request.state.task_id = str(body.task_id)
    with logging_config.bind_task(body.task_id):
        return await _submit(body, request)


async def _submit(body: TaskRequest, request: Request) -> TaskAccepted:
    """Сохранить контекст task_id во всех событиях приёма."""
    broker = request.app.state.task_broker
    settings = request.app.state.settings
    if broker is None:
        raise HTTPException(
            503, "RabbitMQ is unavailable", headers={"Retry-After": retry_after(settings)}
        )
    try:
        return await tasks.submit_task(request.app.state.db_pool, broker, settings, body)
    except tasks.TaskConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (QueueUnavailableError, PostgresError, OSError, TimeoutError) as exc:
        raise HTTPException(
            503,
            "Task was not accepted; retry with the same task_id",
            headers={"Retry-After": retry_after(settings)},
        ) from exc


@router.get("/{task_id}", response_model=TaskResult)
async def status(task_id: UUID, request: Request) -> TaskResult:
    """Получить состояние без обращения к RabbitMQ или MLflow."""
    request.state.task_id = str(task_id)
    try:
        with logging_config.bind_task(task_id):
            result = await tasks.get_task(request.app.state.db_pool, str(task_id))
    except (PostgresError, OSError, TimeoutError) as exc:
        raise HTTPException(503, "PostgreSQL is unavailable") from exc
    if result is None:
        raise HTTPException(404, "Task not found")
    return result
