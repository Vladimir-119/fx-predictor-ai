"""Подготовить JSONL из снимка или последовательно публиковать задачи из файла."""

import argparse
import asyncio
import csv
import json
import logging
import time
from pathlib import Path
from uuid import UUID, uuid4, uuid5

import psutil

from src import logging_config, telemetry
from src.config import get_settings
from src.messaging import connect_broker, create_broker, publish_with_backpressure
from src.ml.data import load_rates
from src.task_schemas import TaskMessage, TaskRequest

log = logging.getLogger(__name__)


def prepare(path: Path, count: int) -> None:
    """Подготовка файла выполняется отдельно от измеряемого источника."""
    rates = load_rates()
    path.parent.mkdir(parents=True, exist_ok=True)
    start = max(19, len(rates) - count)
    with path.open("w", encoding="utf-8") as output:
        for index in range(start, len(rates)):
            history = [
                {"date": str(row.date.date()), "rate": row.rate}
                for row in rates.iloc[index - 19 : index + 1].itertuples()
            ]
            output.write(json.dumps({"history": history}) + "\n")


async def produce(path: Path, output: Path, count: int, job_id: UUID | None = None) -> None:
    """В памяти только одна задача; новый task_id для каждого запуска файла."""
    if path.resolve() in {output.resolve(), output.with_suffix(".job-id").resolve()}:
        raise ValueError("Producer output must not overwrite the input file")
    settings = get_settings()
    logging_config.setup_logging(settings.log_level)
    runtime = telemetry.setup_telemetry(settings)
    broker = create_broker(settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    job_id = job_id or uuid4()
    output.with_suffix(".job-id").write_text(str(job_id) + "\n", encoding="utf-8")
    process = psutil.Process()
    started = time.perf_counter()
    published = 0
    try:
        await connect_broker(broker, settings)
        log.info("source_started", extra={"job_id": str(job_id), "published": 0})
        with (
            path.open(encoding="utf-8") as source,
            output.open("w", newline="", encoding="utf-8") as destination,
        ):
            writer = csv.DictWriter(
                destination,
                fieldnames=[
                    "elapsed",
                    "task_id",
                    "rejections",
                    "rss_bytes",
                ],
            )
            writer.writeheader()
            destination.flush()
            for index, line in enumerate(source):
                if index >= count:
                    break
                payload = json.loads(line)
                payload.pop("task_id", None)
                payload["task_id"] = str(uuid5(job_id, str(index)))
                task = TaskMessage(task=TaskRequest.model_validate(payload))
                # У каждой задачи свой корневой трейс, даже в большой пачке.
                with (
                    logging_config.bind_task(task.task.task_id),
                    telemetry.get_tracer().start_as_current_span("source.task") as span,
                ):
                    span.set_attribute("task.id", str(task.task.task_id))
                    span.set_attribute("job.id", str(job_id))
                    rejected = await publish_with_backpressure(broker, settings, task)
                    published += 1
                writer.writerow(
                    {
                        "elapsed": time.perf_counter() - started,
                        "task_id": str(task.task.task_id),
                        "rejections": rejected,
                        "rss_bytes": process.memory_info().rss,
                    }
                )
                destination.flush()
                if (index + 1) % 20 == 0:
                    log.info(
                        "source_progress", extra={"published": index + 1, "job_id": str(job_id)}
                    )
        log.info("source_completed", extra={"job_id": str(job_id), "published": published})
    finally:
        log.info("graceful_shutdown_started")
        await broker.stop()
        log.info("graceful_shutdown_completed")
        await asyncio.to_thread(runtime.shutdown)


def main() -> None:
    """CLI: --prepare создаёт вход, обычный запуск отправляет его в RabbitMQ."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path("reports/tasks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/producer.csv"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--job-id", type=UUID)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.prepare:
        prepare(args.file, args.count)
    else:
        try:
            asyncio.run(produce(args.file, args.output, args.count, args.job_id))
        except Exception:
            log.exception("source_failed")
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
