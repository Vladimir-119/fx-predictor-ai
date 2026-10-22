"""Четыре воспроизводимых запуска Compose; CSV, графики и выводы для защиты."""

import argparse
import asyncio
import csv
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from uuid import UUID, uuid4

import asyncpg
import httpx
import pandas as pd
from matplotlib import pyplot as plt

import docker
from src.config import get_settings


def compose(arguments: list[str], environment: dict[str, str]) -> None:
    """Без shell: одинаковый вызов на Windows, Linux и macOS."""
    subprocess.run(
        ["docker", "compose", "--profile", "tasks", *arguments], env=environment, check=True
    )


def container_memory(client: docker.DockerClient) -> tuple[int, int]:
    """Одноразовый снимок памяти; producer измеряет собственный RSS отдельно."""
    workers = broker = 0
    for container in client.containers.list(
        filters={"label": "com.docker.compose.project=fx-predictor-ai"}
    ):
        service = container.labels.get("com.docker.compose.service")
        if service in {"rabbitmq", "inference-worker"}:
            stats = client.api.stats(container.id, stream=False, one_shot=True)
            memory = stats["memory_stats"].get("usage", 0)
            if service == "rabbitmq":
                broker += memory
            else:
                workers += memory
    return workers, broker


def read_publications(path: Path) -> list[dict[str, str]]:
    """Источник ещё пишет CSV: незавершённую последнюю строку читаем позже."""
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    complete = content[: content.rfind("\n") + 1]
    return list(csv.DictReader(io.StringIO(complete)))


async def run_case(
    name: str,
    workers: int,
    bounded: bool,
    args: argparse.Namespace,
    output: Path,
    client: docker.DockerClient,
) -> dict[str, str | int | float]:
    settings = get_settings()
    queue = f"{settings.tasks_queue}.bench.{uuid4().hex[:8]}"
    environment = dict(
        os.environ,
        TASKS_QUEUE=queue,
        QUEUE_LIMIT="20" if bounded else "0",
        PREFETCH_COUNT="1" if bounded else "0",
        INFERENCE_DELAY_SECONDS=str(args.delay),
        WORKER_CONCURRENCY="1",
    )
    compose(
        [
            "up",
            "-d",
            "--no-deps",
            "--force-recreate",
            "--wait",
            "--wait-timeout",
            "120",
            "--scale",
            f"inference-worker={workers}",
            "app",
            "inference-worker",
        ],
        environment,
    )
    producer_file = output / f"{name}-producer.csv"
    url = urlsplit(settings.rabbitmq_url.get_secret_value())
    management = f"http://{url.hostname}:{settings.rabbitmq_management_port}"
    auth = (unquote(url.username or ""), unquote(url.password or ""))
    pool = await asyncpg.create_pool(
        settings.database_url.get_secret_value(), min_size=1, max_size=2
    )
    producer = None
    samples = []
    started = time.perf_counter()
    try:
        producer = subprocess.Popen(
            [
                sys.executable,
                "scripts/produce_tasks.py",
                "--file",
                str(args.file),
                "--count",
                str(args.count),
                "--output",
                str(producer_file),
            ],
            env=environment,
        )
        async with httpx.AsyncClient(auth=auth, timeout=10, trust_env=False) as http:
            while time.perf_counter() - started < args.timeout:
                response = await http.get(f"{management}/api/queues/%2F/{quote(queue, safe='')}")
                response.raise_for_status()
                state = response.json()
                worker_memory, broker_memory = await asyncio.to_thread(container_memory, client)
                published = read_publications(producer_file)
                samples.append(
                    {
                        "elapsed": time.perf_counter() - started,
                        "ready": state.get("messages_ready", 0),
                        "unacked": state.get("messages_unacknowledged", 0),
                        "published": len(published),
                        "worker_memory": worker_memory,
                        "broker_memory": broker_memory,
                        "producer_memory": int(published[-1]["rss_bytes"]) if published else 0,
                    }
                )
                if producer.poll() is not None:
                    if producer.returncode:
                        raise RuntimeError(f"Producer failed: {producer.returncode}")
                    ids = [UUID(row["task_id"]) for row in published]
                    completed = await pool.fetchval(
                        "SELECT count(*) FROM tasks WHERE task_id = ANY($1::uuid[]) "
                        "AND status IN ('COMPLETED', 'FAILED')",
                        ids,
                    )
                    if completed == len(ids) and ids:
                        break
                await asyncio.sleep(0.5)
            else:
                raise TimeoutError(f"{name}: tasks did not finish within {args.timeout}s")
        frame = pd.DataFrame(samples)
        frame["source_rate"] = frame.published.diff() / frame.elapsed.diff()
        frame.to_csv(output / f"{name}.csv", index=False)
        rows = await pool.fetch(
            """SELECT status, attempts,
            extract(epoch FROM finished_at-submitted_at)::float AS total,
            extract(epoch FROM finished_at-started_at)::float AS processing
            FROM tasks WHERE task_id = ANY($1::uuid[])""",
            ids,
        )
        durations = pd.DataFrame([dict(row) for row in rows])
        durations.to_csv(output / f"{name}-durations.csv", index=False)
        figure, axes = plt.subplots(2, 2, figsize=(11, 7))
        frame.plot(x="elapsed", y=["ready", "unacked"], ax=axes[0, 0], title=name)
        frame.plot(x="elapsed", y="source_rate", ax=axes[0, 1], title="Accepted tasks / second")
        memory = frame.copy()
        for key in ("producer_memory", "worker_memory", "broker_memory"):
            memory[key] /= 1024**2
        memory.plot(
            x="elapsed",
            y=["producer_memory", "worker_memory", "broker_memory"],
            ax=axes[1, 0],
            title="Memory, MiB",
        )
        durations[["total", "processing"]].plot.hist(
            ax=axes[1, 1], alpha=0.5, title="Task time, seconds"
        )
        figure.tight_layout()
        figure.savefig(output / f"{name}.png")
        plt.close(figure)
        return {
            "case": name,
            "workers": workers,
            "accepted": len(ids),
            "completed": int((durations.status == "COMPLETED").sum()),
            "failed": int((durations.status == "FAILED").sum()),
            "max_ready": int(frame.ready.max()),
            "max_unacked": int(frame.unacked.max()),
            "source_seconds": float(published[-1]["elapsed"]),
            "total_seconds": float(frame.elapsed.iloc[-1]),
            "p95_task_seconds": float(durations.total.quantile(0.95)),
            "rejections": sum(int(row["rejections"]) for row in published),
        }
    finally:
        if producer is not None and producer.poll() is None:
            producer.terminate()
            await asyncio.to_thread(producer.wait, timeout=10)
        await pool.close()


async def main() -> None:
    """Новые очереди каждого запуска сохраняют результаты предыдущих измерений."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path("reports/tasks.jsonl"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.count < 2 or args.delay < 0 or args.timeout <= 0:
        parser.error("Use --count >= 2, --delay >= 0 and --timeout > 0")
    if not args.file.exists():
        raise FileNotFoundError("First run make task-file (requires make dataset)")
    output = Path("docs/backpressure")
    output.mkdir(parents=True, exist_ok=True)
    client = docker.from_env()
    results = []
    try:
        for name, workers, bounded in (
            ("unbounded-1", 1, False),
            ("bounded-1", 1, True),
            ("unbounded-2", 2, False),
            ("bounded-2", 2, True),
        ):
            results.append(await run_case(name, workers, bounded, args, output, client))
        table = pd.DataFrame(results)
        table.to_csv(output / "summary.csv", index=False)
        text = table.to_string(index=False)
        bounded_one = next(row for row in results if row["case"] == "bounded-1")
        bounded_two = next(row for row in results if row["case"] == "bounded-2")
        speedup = bounded_one["total_seconds"] / bounded_two["total_seconds"]
        report = (
            "Реальные измерения последнего запуска `make benchmark`:\n\n```text\n"
            + text
            + "\n```\n\n"
            f"Отношение времени одного worker ко времени двух: {speedup:.2f} "
            "(значение выше 1 означает ускорение) "
            "(сравнивается полное время пакета, включая старт источника и сбор статистики). "
            f"При ограничении очереди максимальный ready составил {bounded_one['max_ready']} "
            f"и {bounded_two['max_ready']} для одного и двух worker. "
            "Скорость источника оценивается по стабильному участку графика "
            "после начального всплеска. "
            "Ready ограничивается очередью, unacked — отдельно через prefetch.\n\n"
            "![One worker](docs/backpressure/bounded-1.png)\n\n"
            "![Two workers](docs/backpressure/bounded-2.png)\n\n"
            "Контрольные запуски без ограничений: "
            "[один worker](docs/backpressure/unbounded-1.png), "
            "[два worker](docs/backpressure/unbounded-2.png).\n"
        )
        readme = Path("README.md")
        content = readme.read_text(encoding="utf-8")
        start = "<!-- BACKPRESSURE_RESULTS_START -->"
        end = "<!-- BACKPRESSURE_RESULTS_END -->"
        if start not in content or end not in content:
            raise ValueError("README measurement markers are missing")
        before, rest = content.split(start, 1)
        _, after = rest.split(end, 1)
        readme.write_text(before + start + "\n" + report + end + after, encoding="utf-8")
        print(text)
        print("Graphs and measurements:", output)
        compose(
            [
                "up",
                "-d",
                "--no-deps",
                "--force-recreate",
                "--wait",
                "--wait-timeout",
                "120",
                "--scale",
                "inference-worker=1",
                "app",
                "inference-worker",
            ],
            dict(os.environ),
        )
        if any(row["failed"] for row in results):
            raise RuntimeError("Some tasks FAILED; inspect durations CSV and worker logs")
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
