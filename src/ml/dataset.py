"""Неизменяемая обучающая выборка: проверенный Kaggle + актуальный PostgreSQL."""

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src import db
from src.config import Settings, get_settings
from src.ml import data
from src.services.rates import expected_latest_date, is_publication_day, update_rates

SNAPSHOT_ROOT = Path("data/snapshots")


@dataclass(frozen=True)
class TrainingSnapshot:
    """Данные и их происхождение всегда передаются вместе."""

    rates: pd.DataFrame
    path: Path
    manifest: dict


def merge_rates(base: pd.DataFrame, recent: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """На пересечении официальный ECB имеет приоритет; будущие даты исключены."""
    combined = pd.concat([base, recent], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="raise")
    combined["rate"] = pd.to_numeric(combined["rate"], errors="raise")
    if combined.date.isna().any() or not np.isfinite(combined.rate).all():
        raise ValueError("Dates and finite rates are required")
    if (combined.rate <= 0).any():
        raise ValueError("Rates must be positive")
    combined = combined.loc[combined.date <= pd.Timestamp(as_of)]
    combined = combined.drop_duplicates("date", keep="last").sort_values("date")
    if combined.empty or combined.date.iloc[-1].date() != as_of:
        raise ValueError(f"Rates are stale: expected {as_of}; run make update-rates")
    known = set(combined.date.dt.date)
    day = base.date.max().date() + timedelta(days=1)
    while day <= as_of:
        if is_publication_day(day) and day not in known:
            raise ValueError(f"Missing ECB publication {day}; run make backfill-rates")
        day += timedelta(days=1)
    return combined[["date", "rate"]].reset_index(drop=True)


def save_snapshot(
    rates: pd.DataFrame, sources: list[dict], root: Path = SNAPSHOT_ROOT
) -> TrainingSnapshot:
    """Адрес зависит от содержимого и происхождения; существующий снимок не меняем."""
    content = rates.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n").encode()
    checksum = hashlib.sha256(content).hexdigest()
    manifest = {
        "sha256": checksum,
        "pair": "USD/EUR",
        "transformation": "EUR per 1 USD; ECB overrides Kaggle on matching dates",
        "first_date": str(rates.date.min().date()),
        "last_date": str(rates.date.max().date()),
        "observations": len(rates),
        "sources": sources,
    }
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["dataset_version"] = identity
    directory = root / identity
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rates.csv"
    metadata = directory / "manifest.json"
    expected = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    for target, value in ((path, content), (metadata, expected.encode())):
        if target.exists():
            if target.read_bytes() != value:
                raise ValueError(f"Snapshot was modified: {target}")
        else:
            with target.open("xb") as handle:
                handle.write(value)
    return load_snapshot(path)


def load_snapshot(path: Path) -> TrainingSnapshot:
    """Offline-повтор не обращается к ECB или PostgreSQL."""
    manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
        raise ValueError("Training snapshot checksum mismatch")
    rates = pd.read_csv(path, parse_dates=["date"], float_precision="round_trip")
    if list(rates.columns) != ["date", "rate"] or manifest["pair"] != "USD/EUR":
        raise ValueError("Unsupported training snapshot schema")
    identity = dict(manifest)
    version = identity.pop("dataset_version")
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != version:
        raise ValueError("Training manifest checksum mismatch")
    return TrainingSnapshot(rates, path, manifest)


async def collect_snapshot(settings: Settings, root: Path = SNAPSHOT_ROOT) -> TrainingSnapshot:
    """Сначала обновить БД, затем прочитать всю доступную историю без LIMIT 60."""
    data.download_snapshot()
    base = data.load_rates()
    cutoff = expected_latest_date(datetime.now(UTC))
    pool = await db.create_db_pool(settings)
    try:
        await update_rates(pool, settings)
        rows = await pool.fetch(
            """SELECT rate_date, value FROM exchange_rates
            WHERE source = 'ecb' AND base = 'EUR' AND quote = 'USD'
              AND rate_type = 'reference' AND rate_date <= $1 ORDER BY rate_date""",
            cutoff,
        )
    finally:
        await db.close_db_pool(pool)
    recent = pd.DataFrame(
        [{"date": row["rate_date"], "rate": 1.0 / row["value"]} for row in rows],
        columns=["date", "rate"],
    )
    rates = merge_rates(base, recent, cutoff)
    sources = [
        {"kind": "kaggle", **data.read_manifest()},
        {"kind": "ecb", "source": settings.ecb_history_url, "table": "exchange_rates"},
    ]
    return save_snapshot(rates, sources, root)


def main() -> None:
    """Создать снимок без обучения или проверить сохранённый."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    snapshot = (
        load_snapshot(args.snapshot)
        if args.snapshot
        else asyncio.run(collect_snapshot(get_settings()))
    )
    print(snapshot.path)
    print(json.dumps(snapshot.manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
