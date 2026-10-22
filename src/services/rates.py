"""Загрузка ECB, публикационный календарь и история в PostgreSQL."""

import asyncio
import logging
from datetime import date, datetime, time, timedelta
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import asyncpg
import httpx
import pandas as pd
from dateutil.easter import easter

from src import db, logging_config
from src.config import Settings, get_settings
from src.ml.data import RAW_PATH, verify_snapshot
from src.schemas import DisplayRate, RateObservation, RatesResponse

log = logging.getLogger(__name__)
UPSERT_RATE = """
INSERT INTO exchange_rates (source, base, quote, rate_type, rate_date, value)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (source, base, quote, rate_type, rate_date)
DO UPDATE SET value = EXCLUDED.value, fetched_at = now()
"""


def is_publication_day(day: date) -> bool:
    """Учесть выходные и TARGET-праздники современного календаря ECB."""
    holidays = {
        date(day.year, 1, 1),
        date(day.year, 5, 1),
        date(day.year, 12, 25),
        date(day.year, 12, 26),
        easter(day.year) - timedelta(days=2),
        easter(day.year) + timedelta(days=1),
    }
    return day.weekday() < 5 and day not in holidays


def next_publication_date(day: date) -> date:
    """Найти следующий день публикации вместо механического date + 1."""
    result = day + timedelta(days=1)
    while not is_publication_day(result):
        result += timedelta(days=1)
    return result


def expected_latest_date(now: datetime) -> date:
    """Дать ECB 30 минут после обычной публикации в 16:00 по Европе."""
    local = now.astimezone(ZoneInfo("Europe/Berlin"))
    result = local.date()
    if local.time() < time(16, 30):
        result -= timedelta(days=1)
    while not is_publication_day(result):
        result -= timedelta(days=1)
    return result


def parse_ecb_xml(content: bytes) -> list[RateObservation]:
    """Прочитать опубликованные даты и котировки из официального XML."""
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        raise ValueError("ECB returned invalid XML") from exc
    observations = []
    for daily in root.iter():
        if "time" not in daily.attrib:
            continue
        for currency in daily:
            observations.append(
                RateObservation(
                    base="EUR",
                    quote=currency.attrib["currency"],
                    rate_date=date.fromisoformat(daily.attrib["time"]),
                    value=float(currency.attrib["rate"]),
                )
            )
    if not observations:
        raise ValueError("ECB returned no exchange rates")
    keys = [(row.rate_date, row.quote) for row in observations]
    if len(keys) != len(set(keys)):
        raise ValueError("ECB returned duplicate rates")
    return observations


async def save_rates(pool: asyncpg.Pool, observations: list[RateObservation]) -> int:
    """Сохранить таблицу атомарно, повторная загрузка не создаёт дубликаты."""
    values = [
        (row.source, row.base, row.quote, row.rate_type, row.rate_date, row.value)
        for row in observations
    ]
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.executemany(UPSERT_RATE, values)
    return len(values)


async def update_rates(pool: asyncpg.Pool, settings: Settings) -> int:
    """Загрузить последние 90 дней и восполнить пропуски после остановки."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(settings.ecb_history_url)
    response.raise_for_status()
    observations = parse_ecb_xml(response.content)
    return await save_rates(pool, observations)


async def import_snapshot(pool: asyncpg.Pool) -> int:
    """Загрузить последние 120 календарных дней снимка для запуска API."""
    verify_snapshot()
    raw = pd.read_csv(RAW_PATH)
    cutoff = pd.to_datetime(raw.date).max() - pd.Timedelta(days=120)
    recent = raw.loc[(pd.to_datetime(raw.date) >= cutoff) & (raw.type == "reference")]
    observations = [
        RateObservation(
            base=row.base, quote=row.quote, rate_date=date.fromisoformat(row.date), value=row.value
        )
        for row in recent.itertuples(index=False)
    ]
    return await save_rates(pool, observations)


async def load_history(pool: asyncpg.Pool | asyncpg.Connection, as_of: date) -> pd.DataFrame:
    """Получить только известные на указанную дату значения USD/EUR."""
    rows = await pool.fetch(
        """SELECT rate_date, value FROM exchange_rates
        WHERE source = 'ecb' AND base = 'EUR' AND quote = 'USD'
          AND rate_type = 'reference' AND rate_date <= $1
        ORDER BY rate_date DESC LIMIT 60""",
        as_of,
    )
    frame = pd.DataFrame(
        [{"date": row["rate_date"], "rate": 1.0 / row["value"]} for row in rows],
        columns=["date", "rate"],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


async def latest_rates(pool: asyncpg.Pool, now: datetime) -> RatesResponse:
    """Вернуть одну опубликованную таблицу, нормализованную через USD."""
    rows = await pool.fetch(
        """SELECT quote, value, rate_date FROM exchange_rates
        WHERE source = 'ecb' AND base = 'EUR' AND rate_type = 'reference'
          AND rate_date = (SELECT max(rate_date) FROM exchange_rates
              WHERE source = 'ecb' AND base = 'EUR' AND rate_type = 'reference'
                AND rate_date <= $1)
        ORDER BY quote""",
        now.date(),
    )
    if not rows:
        raise ValueError("Rates are missing; run make update-rates")
    dollar = next((row["value"] for row in rows if row["quote"] == "USD"), None)
    if dollar is None:
        raise ValueError("ECB table has no USD quote")
    displayed = [DisplayRate(quote="EUR", value=1.0 / dollar)]
    displayed.extend(DisplayRate(quote=row["quote"], value=row["value"] / dollar) for row in rows)
    return RatesResponse(
        rate_date=rows[0]["rate_date"],
        stale=rows[0]["rate_date"] < expected_latest_date(now),
        rates=displayed,
    )


async def run_worker(pool: asyncpg.Pool, settings: Settings) -> None:
    """Обновлять данные независимо от API и повторять после сбоя источника."""
    while True:
        try:
            count = await update_rates(pool, settings)
        except (httpx.HTTPError, ValueError, KeyError, asyncpg.PostgresError, OSError) as exc:
            log.warning("rate_update_failed", extra={"error_type": type(exc).__name__})
        else:
            log.info("rates_updated", extra={"observations": count})
        await asyncio.sleep(settings.update_interval_seconds)


async def main() -> None:
    """Запустить выбранную команду данных: import, update или worker."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["import", "update", "worker"])
    command = parser.parse_args().command
    settings = get_settings()
    logging_config.setup_logging(settings.log_level)
    pool = await db.create_db_pool(settings)
    try:
        if command == "worker":
            await run_worker(pool, settings)
        elif command == "import":
            print(f"Imported {await import_snapshot(pool)} observations")
        else:
            print(f"Updated {await update_rates(pool, settings)} observations")
    finally:
        await db.close_db_pool(pool)


if __name__ == "__main__":
    asyncio.run(main())
