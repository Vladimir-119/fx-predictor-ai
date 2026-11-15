"""Проверки формата пакетного источника и чтения статистики во время записи."""

import csv
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pandas as pd

from scripts import benchmark, produce_tasks
from src.task_schemas import TaskRequest


async def test_prepared_windows_and_restart_keep_stable_task_ids(tmp_path, monkeypatch):
    rates = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=30), "rate": 0.8})
    monkeypatch.setattr(produce_tasks, "load_rates", lambda: rates)
    source = tmp_path / "tasks.jsonl"
    produce_tasks.prepare(source, 3)
    inputs = [TaskRequest.model_validate_json(line) for line in source.read_text().splitlines()]
    assert all(set(json.loads(line)) == {"history"} for line in source.read_text().splitlines())
    assert len(inputs) == 3
    assert all(len(task.history) == 20 for task in inputs)
    assert inputs[-1].history[-1].date == rates.date.iloc[-1].date()
    assert inputs[0].history[1:] == inputs[1].history[:-1]

    broker = SimpleNamespace(stop=AsyncMock())
    sender = AsyncMock(return_value=2)
    monkeypatch.setattr(produce_tasks, "create_broker", lambda settings: broker)
    monkeypatch.setattr(produce_tasks, "connect_broker", AsyncMock())
    monkeypatch.setattr(produce_tasks, "publish_with_backpressure", sender)
    output = tmp_path / "producer.csv"
    job_id = uuid4()
    await produce_tasks.produce(source, output, 2, job_id)
    first_ids = [call.args[-1].task.task_id for call in sender.call_args_list]
    sender.reset_mock()
    await produce_tasks.produce(source, output, 2, job_id)
    assert [call.args[-1].task.task_id for call in sender.call_args_list] == first_ids
    sender.reset_mock()
    await produce_tasks.produce(source, output, 2)
    assert set(first_ids).isdisjoint(call.args[-1].task.task_id for call in sender.call_args_list)
    with output.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 2
    assert all(int(row["rejections"]) == 2 and int(row["rss_bytes"]) > 0 for row in records)
    assert output.with_suffix(".job-id").read_text().strip()
    assert broker.stop.await_count == 3


def test_benchmark_ignores_a_partially_written_csv_row(tmp_path):
    path = tmp_path / "producer.csv"
    assert benchmark.read_publications(path) == []
    path.write_text("elapsed,task_id,rss_bytes\n0.1,complete,100\n0.2,incompl", encoding="utf-8")
    assert benchmark.read_publications(path) == [
        {"elapsed": "0.1", "task_id": "complete", "rss_bytes": "100"}
    ]
