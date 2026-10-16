"""Ручные примеры данных и проверка отсутствия утечки."""

import hashlib
import json

import httpx
import numpy as np
import pandas as pd
import pytest

from src.ml import data, features


def example_rates(size=160):
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=size),
            "rate": 0.8 + np.arange(size) / 1000,
        }
    )


def test_dataset_download_checksum_currency_direction_and_validation(tmp_path, monkeypatch):
    content = b"date,base,quote,type,value\n2026-10-07,EUR,USD,reference,1.2\n"
    raw_path = tmp_path / "raw/ecb.csv"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "download_url": "https://example.test/ecb.csv",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(data, "RAW_PATH", raw_path)
    monkeypatch.setattr(data, "MANIFEST_PATH", manifest)
    calls = []

    def download(url, **kwargs):
        calls.append(url)
        return httpx.Response(200, content=content, request=httpx.Request("GET", url))

    monkeypatch.setattr(data.httpx, "get", download)
    assert data.download_snapshot() == raw_path
    assert data.download_snapshot() == raw_path
    assert len(calls) == 1
    assert data.load_rates().rate.iloc[0] == pytest.approx(1 / 1.2)
    raw_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        data.download_snapshot()
    raw_path.unlink()
    manifest.write_text(
        json.dumps({"download_url": "https://example.test/ecb.csv", "sha256": "wrong"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum"):
        data.download_snapshot()
    assert not raw_path.exists()

    raw = pd.DataFrame(
        [
            {
                "date": "2026-10-07",
                "base": "EUR",
                "quote": "USD",
                "type": "reference",
                "value": 1.25,
            },
            {
                "date": "2026-10-06",
                "base": "EUR",
                "quote": "USD",
                "type": "reference",
                "value": 1.2,
            },
            {
                "date": "2026-10-07",
                "base": "EUR",
                "quote": "JPY",
                "type": "reference",
                "value": 180,
            },
        ]
    )
    result = data.prepare_rates(raw)
    assert result.rate.tolist() == pytest.approx([1 / 1.2, 0.8])
    assert result.date.is_monotonic_increasing
    for bad in (
        raw.drop(columns="type"),
        raw[raw.quote == "JPY"],
        pd.concat([raw, raw.iloc[[0]]]),
        raw.assign(value=0),
        raw.assign(value=np.inf),
        raw.assign(date=None),
    ):
        with pytest.raises(ValueError):
            data.prepare_rates(bad)


def test_features_match_manual_values_and_future_changes_do_not_leak():
    rates = example_rates()
    frame = features.make_training_frame(rates)
    first = frame.iloc[0]
    assert first.rate == pytest.approx(0.819)
    assert first.lag_1 == pytest.approx(0.818)
    assert first.mean_20 == pytest.approx(rates.rate.iloc[:20].mean())
    assert first.target_change == pytest.approx(0.001)
    assert first.target_date == rates.date.iloc[20]
    changed = rates.copy()
    changed.loc[80:, "rate"] = 999
    pd.testing.assert_frame_equal(
        features.build_features(rates).iloc[:80], features.build_features(changed).iloc[:80]
    )
    pd.testing.assert_frame_equal(
        features.latest_features(rates.iloc[:80]),
        features.build_features(rates).iloc[[79]][features.FEATURE_COLUMNS],
    )
    training, validation, holdout = features.split_by_time(frame)
    assert training.target_date.max() <= validation.date.min()
    assert validation.target_date.max() <= holdout.date.min()
    assert len(training) + len(validation) + len(holdout) == len(frame)
    for invalid in (rates.iloc[::-1], pd.concat([rates, rates.iloc[[-1]]])):
        with pytest.raises(ValueError, match="sorted"):
            features.build_features(invalid)
    with pytest.raises(ValueError, match="20"):
        features.latest_features(rates.head(19))
    with pytest.raises(ValueError, match="100"):
        features.split_by_time(frame.head(99))
