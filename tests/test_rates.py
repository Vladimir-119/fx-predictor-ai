"""Календарь публикаций, парсинг источника и повторное обновление."""

import asyncio
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pandas as pd
import pytest

from src.services import rates

XML = b"""<Envelope><Cube><Cube time="2026-10-07">
<Cube currency="USD" rate="1.2"/><Cube currency="JPY" rate="180"/>
</Cube></Cube></Envelope>"""


def test_calendar_handles_weekends_target_holidays_and_publication_cutoff():
    assert rates.next_publication_date(date(2026, 10, 9)) == date(2026, 10, 12)
    assert rates.next_publication_date(date(2026, 4, 2)) == date(2026, 4, 7)
    assert rates.next_publication_date(date(2026, 12, 24)) == date(2026, 12, 28)
    assert rates.expected_latest_date(datetime(2026, 10, 8, 12, tzinfo=UTC)) == date(2026, 10, 7)
    assert rates.expected_latest_date(datetime(2026, 10, 8, 15, tzinfo=UTC)) == date(2026, 10, 8)
    assert rates.expected_latest_date(datetime(2026, 10, 11, 15, tzinfo=UTC)) == date(2026, 10, 9)


async def test_source_is_validated_and_imports_are_atomic(settings, monkeypatch, tmp_path):
    rows = rates.parse_ecb_xml(XML)
    assert len(rows) == 2
    assert rows[0].rate_date == date(2026, 10, 7)
    assert rows[0].base == "EUR" and rows[0].quote == "USD"
    for invalid in (
        b"not XML",
        b"<Envelope/>",
        XML.replace(b"1.2", b"0"),
        XML.replace(b"JPY", b"USD"),
    ):
        with pytest.raises(ValueError):
            rates.parse_ecb_xml(invalid)

    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=XML))
    original = httpx.AsyncClient
    monkeypatch.setattr(rates.httpx, "AsyncClient", lambda **kwargs: original(transport=transport))
    connection = AsyncMock()
    connection.transaction = MagicMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = connection
    assert await rates.update_rates(pool, settings) == 2
    connection.executemany.assert_awaited_once()
    sql, values = connection.executemany.call_args.args
    assert "ON CONFLICT" in sql
    assert values[0] == ("ecb", "EUR", "USD", "reference", date(2026, 10, 7), 1.2)
    snapshot = tmp_path / "ecb.csv"
    pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "base": "EUR",
                "quote": "USD",
                "type": "reference",
                "value": 1.1,
            },
            {
                "date": "2026-10-07",
                "base": "EUR",
                "quote": "USD",
                "type": "reference",
                "value": 1.2,
            },
        ]
    ).to_csv(snapshot, index=False)
    monkeypatch.setattr(rates, "RAW_PATH", snapshot)
    monkeypatch.setattr(rates, "verify_snapshot", MagicMock())
    assert await rates.import_snapshot(pool) == 1
    assert connection.executemany.call_args.args[1][0][-1] == 1.2


async def test_rates_are_normalized_and_history_is_sorted(pool):
    now = datetime(2026, 10, 8, 12, tzinfo=UTC)
    pool.fetch.return_value = [
        {"quote": "USD", "value": 1.2, "rate_date": date(2026, 10, 7)},
        {"quote": "JPY", "value": 180, "rate_date": date(2026, 10, 7)},
    ]
    result = await rates.latest_rates(pool, now)
    assert not result.stale
    assert {row.quote: row.value for row in result.rates} == pytest.approx(
        {"EUR": 1 / 1.2, "USD": 1, "JPY": 150}
    )
    pool.fetch.return_value = [
        {"rate_date": date(2026, 10, 7), "value": 1.2},
        {"rate_date": date(2026, 10, 6), "value": 1.25},
    ]
    history = await rates.load_history(pool, now.date())
    assert history.rate.tolist() == pytest.approx([0.8, 1 / 1.2])
    pool.fetch.return_value = []
    assert (await rates.load_history(pool, now.date())).empty
    with pytest.raises(ValueError, match="missing"):
        await rates.latest_rates(pool, now)
    pool.fetch.return_value = [{"quote": "JPY", "value": 180, "rate_date": now.date()}]
    with pytest.raises(ValueError, match="USD"):
        await rates.latest_rates(pool, now)


async def test_worker_retries_after_source_failure(settings, pool, monkeypatch):
    updater = AsyncMock(side_effect=[httpx.ConnectError("offline"), 2])
    sleeper = AsyncMock(side_effect=[None, asyncio.CancelledError()])
    monkeypatch.setattr(rates, "update_rates", updater)
    monkeypatch.setattr(rates.asyncio, "sleep", sleeper)
    with pytest.raises(asyncio.CancelledError):
        await rates.run_worker(pool, settings)
    assert updater.await_count == 2
    assert sleeper.await_count == 2
