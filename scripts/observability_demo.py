"""Отправить обычные, ошибочные или конкурентные запросы для стенда наблюдаемости."""

import argparse
import asyncio
import logging
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx

from src.config import get_settings
from src.logging_config import setup_logging

log = logging.getLogger(__name__)


def task_payload(poison: bool = False) -> dict[str, object]:
    """Синтетическая история проверяет конвейер, а не качество прогноза модели."""
    day = datetime.now(UTC).date() - timedelta(days=1)
    dates = []
    while len(dates) < 20:
        if day.weekday() < 5:
            dates.append(day)
        day -= timedelta(days=1)
    dates.reverse()
    payload: dict[str, object] = {
        "task_id": str(uuid4()),
        "history": [
            {"date": day.isoformat(), "rate": 0.85 + index * 0.0001}
            for index, day in enumerate(dates)
        ],
    }
    if poison:
        # Схема валидна, но inference требует следующую публикацию после истории.
        payload["forecast_date"] = dates[-1].isoformat()
    return payload


async def wait_for_task(
    client: httpx.AsyncClient, task_id: str, timeout: float
) -> dict[str, object]:
    """Дождаться окончательного статуса, не скрывая зависшую или потерянную задачу."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get(f"/api/v1/tasks/{task_id}")
        response.raise_for_status()
        result = response.json()
        if result["status"] in {"COMPLETED", "FAILED"}:
            log.info(
                "demo_task_finished",
                extra={
                    "task_id": task_id,
                    "result": result["status"],
                    "attempt": result["attempts"],
                    "model_version": (result.get("result") or {}).get("model", {}).get("version"),
                },
            )
            return result
        await asyncio.sleep(0.5)
    raise TimeoutError(f"Task {task_id} did not finish within {timeout:g} seconds")


async def run_demo(
    mode: str, base_url: str, count: int, concurrency: int, timeout: float, wait: bool
) -> None:
    """Не повторять 503: этот сценарий специально показывает отказ HTTP при перегрузке."""
    slots = asyncio.Semaphore(concurrency)
    statuses: Counter[int] = Counter()
    accepted: list[str] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=15, trust_env=False) as client:
        (await client.get("/healthz")).raise_for_status()

        async def submit() -> None:
            payload = task_payload(poison=mode == "poison")
            async with slots:
                response = await client.post("/api/v1/tasks", json=payload)
            statuses[response.status_code] += 1
            if response.status_code == 202:
                accepted.append(str(payload["task_id"]))
            elif response.status_code != 503:
                response.raise_for_status()
                raise RuntimeError(f"Unexpected response status: {response.status_code}")
            log.info(
                "demo_request_completed",
                extra={
                    "task_id": payload["task_id"],
                    "status_code": response.status_code,
                    "returned_trace_id": response.headers.get("X-Trace-ID"),
                },
            )

        await asyncio.gather(*(submit() for _ in range(count)))
        log.info(
            "demo_batch_submitted: mode=%s sent=%s accepted=%s unavailable=%s",
            mode,
            count,
            statuses[202],
            statuses[503],
        )
        if not accepted:
            raise RuntimeError("No task was accepted; check queue, broker and worker health")
        if wait:
            for task_id in accepted:
                result = await wait_for_task(client, task_id, timeout)
                expected = "FAILED" if mode == "poison" else "COMPLETED"
                if result["status"] != expected:
                    raise RuntimeError(
                        f"Task {task_id}: expected {expected}, got {result['status']}"
                    )
        if mode == "overload" and not statuses[503]:
            log.warning("demo_no_overload: increase --count or --concurrency to observe HTTP 503")


def main() -> None:
    """Нагрузка запускается с ноутбука; её API-запросы видны в Docker-сервисах."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("normal", "overload", "poison"), default="normal")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--wait", action="store_true", help="Wait for all accepted tasks")
    args = parser.parse_args()
    count = args.count if args.count is not None else (300 if args.mode == "overload" else 1)
    if count < 1 or args.concurrency < 1 or args.timeout <= 0:
        parser.error("--count, --concurrency and --timeout must be positive")
    setup_logging(get_settings().log_level)
    asyncio.run(
        run_demo(
            args.mode,
            args.base_url,
            count,
            args.concurrency,
            args.timeout,
            args.wait or args.mode != "overload",
        )
    )


if __name__ == "__main__":
    main()
