"""Свежие данные, воспроизводимость, общие окна и явное продвижение модели."""

import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.ml import dataset, train
from src.ml.features import FEATURE_COLUMNS, make_training_frame
from src.ml.models import Candidate, ExperimentConfig, create_model, read_config
from src.services import rates


def history(size=160):
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=size),
            "rate": 0.8 + np.sin(np.arange(size)) / 1000,
        }
    )


def small_config():
    return ExperimentConfig(
        validation_size=30,
        holdout_size=15,
        min_training_size=40,
        candidates=[
            Candidate(name="naive", model="baseline"),
            Candidate(name="ridge", model="ridge"),
        ],
    )


def test_merge_prefers_official_values_excludes_future_and_rejects_gaps():
    base = pd.DataFrame({"date": pd.to_datetime(["2026-10-06", "2026-10-07"]), "rate": [0.8, 0.81]})
    recent = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-07", "2026-10-08", "2026-10-09", "2026-10-12"]),
            "rate": [0.82, 0.83, 0.84, 0.85],
        }
    )
    result = dataset.merge_rates(base, recent, date(2026, 10, 9))
    assert result.rate.tolist() == [0.8, 0.82, 0.83, 0.84]
    pd.testing.assert_frame_equal(dataset.merge_rates(result, recent, date(2026, 10, 9)), result)
    with pytest.raises(ValueError, match="Missing ECB publication"):
        dataset.merge_rates(base, recent.drop(index=1), date(2026, 10, 9))
    with pytest.raises(ValueError, match="stale"):
        dataset.merge_rates(base, recent, date(2026, 10, 13))
    for value in (0, np.nan, np.inf):
        with pytest.raises(ValueError):
            dataset.merge_rates(base, recent.assign(rate=value), date(2026, 10, 9))


def test_snapshot_is_immutable_repeatable_and_detects_modified_data_and_lineage(tmp_path):
    frame = history()
    sources = [{"kind": "test", "source": "https://example.test/rates"}]
    snapshot = dataset.save_snapshot(frame, sources, tmp_path)
    assert dataset.save_snapshot(frame, sources, tmp_path).path == snapshot.path
    pd.testing.assert_frame_equal(snapshot.rates, frame)
    original = snapshot.path.read_bytes()
    snapshot.path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        dataset.load_snapshot(snapshot.path)
    with pytest.raises(ValueError, match="modified"):
        dataset.save_snapshot(frame, sources, tmp_path)
    snapshot.path.write_bytes(original)
    manifest = snapshot.path.with_name("manifest.json")
    saved = manifest.read_text(encoding="utf-8")
    altered = json.loads(saved)
    altered["sources"] = []
    manifest.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest checksum"):
        dataset.load_snapshot(snapshot.path)
    with pytest.raises(ValueError, match="modified"):
        dataset.save_snapshot(frame, sources, tmp_path)
    altered["pair"] = "EUR/JPY"
    manifest.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        dataset.load_snapshot(snapshot.path)


async def test_collection_updates_database_and_never_silently_falls_back(
    settings, tmp_path, monkeypatch
):
    base = pd.DataFrame({"date": pd.to_datetime(["2026-10-06", "2026-10-07"]), "rate": [0.8, 0.81]})
    pool = AsyncMock()
    pool.fetch.return_value = [
        {"rate_date": date(2026, 10, 8), "value": 1.25},
        {"rate_date": date(2026, 10, 9), "value": 1.2},
    ]
    monkeypatch.setattr(dataset.data, "download_snapshot", MagicMock())
    monkeypatch.setattr(dataset.data, "load_rates", lambda: base)
    monkeypatch.setattr(
        dataset.data,
        "read_manifest",
        lambda: {"source": "https://example.test/kaggle", "sha256": "base-hash"},
    )
    monkeypatch.setattr(dataset, "expected_latest_date", lambda now: date(2026, 10, 9))
    monkeypatch.setattr(dataset.db, "create_db_pool", AsyncMock(return_value=pool))
    updater = AsyncMock()
    monkeypatch.setattr(dataset, "update_rates", updater)
    snapshot = await dataset.collect_snapshot(settings, tmp_path)
    assert snapshot.rates.rate.iloc[-1] == pytest.approx(1 / 1.2)
    assert snapshot.manifest["sources"][1]["table"] == "exchange_rates"
    assert "LIMIT" not in pool.fetch.call_args.args[0]
    assert settings.database_url.get_secret_value() not in json.dumps(snapshot.manifest)
    updater.assert_awaited_once_with(pool, settings)
    pool.close.assert_awaited_once()
    updater.side_effect = OSError("source unavailable")
    with pytest.raises(OSError):
        await dataset.collect_snapshot(settings, tmp_path)
    assert pool.close.await_count == 2


def test_config_validation_and_model_defaults(tmp_path):
    config = read_config(Path("configs/experiments.toml"))
    assert len(config.candidates) == 6
    assert create_model(config.candidates[2]).get_params()["loss_function"] == "MAE"
    assert create_model(config.candidates[-1]).get_params()["loss_function"] == "RMSE"
    assert create_model(config.candidates[3]).get_params()["depth"] == 3
    assert create_model(config.candidates[1]).get_params()["ridge__alpha"] == 10
    for change in (
        {"candidates": [config.candidates[0], config.candidates[0]]},
        {"candidates": config.candidates[1:]},
        {"validation_size": 3, "cv_splits": 4},
        {"feature_set": "news-v2"},
        {"typo": 1},
        {
            "candidates": [
                {"name": "naive", "model": "baseline", "params": {"constant": 1}},
                config.candidates[1].model_dump(),
            ]
        },
        {"candidates": [{"name": "wrong", "model": "unknown"}, config.candidates[0].model_dump()]},
    ):
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate({**config.model_dump(), **change})
    path = tmp_path / "experiments.toml"
    path.write_text('feature_set = "новости"', encoding="utf-8")
    with pytest.raises(ValueError):
        read_config(path)


def test_time_windows_backtest_and_metrics_have_no_future_data():
    frame = make_training_frame(history(1700))
    config = small_config()
    training, validation, holdout = train.split_frame(frame, config)
    assert training.target_date.max() <= validation.date.min()
    assert validation.target_date.max() <= holdout.date.min()
    limited = Candidate(name="recent", model="ridge", train_years=1)
    recent = train.training_window(training, limited, 40)
    assert 200 < len(recent) < 270
    assert recent.target_date.max() == training.target_date.max()
    assert recent.target_date.min() > training.target_date.max() - pd.DateOffset(years=1)
    folds = train.backtest(limited, pd.concat([training, validation]), config)
    assert len(folds) == 3
    assert (folds.train_target_end < folds.validation_start).all()
    assert folds.validation_end.max() < str(holdout.target_date.min().date())
    metrics = train.calculate_metrics(validation, validation.rate.to_numpy())
    assert metrics["skill_vs_naive"] == 0
    assert metrics["mae"] == metrics["naive_mae"]
    unchanged = validation.assign(target_rate=validation.rate, target_change=0)
    assert "skill_vs_naive" not in train.calculate_metrics(unchanged, unchanged.rate.to_numpy())
    with pytest.raises(ValueError, match="examples"):
        train.split_frame(frame.head(10), config)
    with pytest.raises(ValueError, match="training rows"):
        train.training_window(training, limited, 500)


def test_promotion_rejects_incompatible_model_without_switching_alias(settings, monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(train, "MlflowClient", lambda: client)
    loader = MagicMock(return_value=SimpleNamespace(metadata=SimpleNamespace(metadata={})))
    monkeypatch.setattr(train.mlflow.pyfunc, "load_model", loader)
    valid = {"pair": "USD/EUR", "target": "next_rate_change", "features": FEATURE_COLUMNS}
    for invalid in (
        {},
        {**valid, "target": "price"},
        {**valid, "features": []},
        {**valid, "feature_set": "news-v2"},
    ):
        loader.return_value.metadata.metadata = invalid
        with pytest.raises(ValueError, match="incompatible"):
            train.promote(settings, "1")
    client.set_registered_model_alias.assert_not_called()


def test_training_cli_supports_offline_replay_fresh_collection_and_explicit_promotion(
    settings, tmp_path, monkeypatch
):
    snapshot = dataset.save_snapshot(history(), [{"kind": "test"}], tmp_path / "snapshots")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "get_settings", lambda: settings)
    monkeypatch.setattr(train, "read_config", lambda path: small_config())
    worker = MagicMock(return_value={"candidate_version": "1"})
    collector = AsyncMock(return_value=snapshot)
    promoter = MagicMock()
    monkeypatch.setattr(train, "train", worker)
    monkeypatch.setattr(train, "collect_snapshot", collector)
    monkeypatch.setattr(train, "promote", promoter)
    monkeypatch.setattr(sys, "argv", ["train", "--snapshot", str(snapshot.path)])
    train.main()
    collector.assert_not_awaited()
    assert json.loads(Path("reports/training.json").read_text())["candidate_version"] == "1"
    monkeypatch.setattr(sys, "argv", ["train"])
    train.main()
    collector.assert_awaited_once()
    monkeypatch.setattr(sys, "argv", ["train", "--promote", "2"])
    train.main()
    promoter.assert_called_once_with(settings, "2")
    for invalid in ("", "0"):
        monkeypatch.setattr(sys, "argv", ["train", "--promote", invalid])
        with pytest.raises(SystemExit):
            train.main()
    monkeypatch.setattr(dataset, "get_settings", lambda: settings)
    monkeypatch.setattr(dataset, "collect_snapshot", collector)
    monkeypatch.setattr(sys, "argv", ["dataset", "--snapshot", str(snapshot.path)])
    dataset.main()
    monkeypatch.setattr(sys, "argv", ["dataset"])
    dataset.main()


async def test_backfill_uses_full_source_and_closes_pool_on_failure(settings, monkeypatch):
    pool = AsyncMock()
    updater = AsyncMock(return_value=5)
    monkeypatch.setattr(rates, "get_settings", lambda: settings)
    monkeypatch.setattr(rates.db, "create_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(rates, "update_rates", updater)
    monkeypatch.setattr(rates, "import_snapshot", AsyncMock(return_value=2))
    monkeypatch.setattr(rates, "run_worker", AsyncMock())
    for command in ("import", "update", "backfill", "worker"):
        monkeypatch.setattr(sys, "argv", ["rates", command])
        await rates.main()
        if command == "backfill":
            assert updater.call_args.args[1].ecb_history_url == settings.ecb_full_history_url
    updater.side_effect = OSError("offline")
    monkeypatch.setattr(sys, "argv", ["rates", "update"])
    with pytest.raises(OSError):
        await rates.main()
    assert pool.close.await_count == 5
