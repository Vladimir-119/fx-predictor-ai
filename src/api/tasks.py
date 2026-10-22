"""API приёма задач: перегруженный брокер означает быстрый отказ."""

from uuid import UUID

from asyncpg import PostgresError
from fastapi import APIRouter, HTTPException, Request

from src.messaging import QueueUnavailableError, retry_after
from src.services import tasks
from src.task_schemas import TaskAccepted, TaskRequest, TaskResult

router = APIRouter(prefix="/api/v1/tasks", tags=["Tasks"])


@router.post("", response_model=TaskAccepted, status_code=202)
async def submit(body: TaskRequest, request: Request) -> TaskAccepted:
    """Принять задачу; при перегрузке не ждать свободного места в очереди."""
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
    try:
        result = await tasks.get_task(request.app.state.db_pool, str(task_id))
    except (PostgresError, OSError, TimeoutError) as exc:
        raise HTTPException(503, "PostgreSQL is unavailable") from exc
    if result is None:
        raise HTTPException(404, "Task not found")
    return result
