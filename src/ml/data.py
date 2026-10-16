"""Неизменяемый снимок ECB из Kaggle и единое направление USD → EUR."""

import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

MANIFEST_PATH = Path("data/manifest.json")
RAW_PATH = Path("data/raw/ecb.csv")


def read_manifest() -> dict[str, str | int]:
    """Прочитать описание зафиксированного снимка."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_snapshot(path: Path | None = None) -> None:
    """Проверить содержимое файла до анализа или обучения."""
    path = path or RAW_PATH
    expected = read_manifest()["sha256"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Dataset checksum differs from data/manifest.json")


def download_snapshot() -> Path:
    """Скачать только закреплённую версию, не перезаписывая другой снимок."""
    if RAW_PATH.exists():
        verify_snapshot()
        return RAW_PATH
    manifest = read_manifest()
    response = httpx.get(str(manifest["download_url"]), timeout=120, follow_redirects=True)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != manifest["sha256"]:
        raise ValueError("Downloaded dataset has an unexpected checksum")
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_bytes(response.content)
    return RAW_PATH


def prepare_rates(raw: pd.DataFrame) -> pd.DataFrame:
    """Выделить опубликованный ECB ряд и перевести его в EUR за один USD."""
    required = {"date", "base", "quote", "type", "value"}
    if not required.issubset(raw.columns):
        raise ValueError("Expected date, base, quote, type, value columns")
    selected = raw.loc[
        (raw["base"] == "EUR") & (raw["quote"] == "USD") & (raw["type"] == "reference"),
        ["date", "value"],
    ].copy()
    if selected.empty:
        raise ValueError("ECB EUR/USD reference series is missing")
    selected["date"] = pd.to_datetime(selected["date"], format="%Y-%m-%d", errors="raise")
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    if not np.isfinite(selected["value"]).all() or (selected["value"] <= 0).any():
        raise ValueError("Rates must be finite and positive")
    if selected["date"].isna().any() or selected["date"].duplicated().any():
        raise ValueError("Dates must be present and unique")
    selected["rate"] = 1.0 / selected["value"]
    return selected[["date", "rate"]].sort_values("date").reset_index(drop=True)


def load_rates() -> pd.DataFrame:
    """Вернуть проверенный ряд для EDA и обучения."""
    verify_snapshot()
    return prepare_rates(pd.read_csv(RAW_PATH))


def main() -> None:
    """Скачать снимок и вывести границы выбранного ряда."""
    download_snapshot()
    rates = load_rates()
    print(f"USD/EUR: {len(rates)} observations, {rates.date.min()} → {rates.date.max()}")


if __name__ == "__main__":
    main()
