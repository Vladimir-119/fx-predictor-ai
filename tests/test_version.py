"""Версия API и версия приложения имеют разный смысл."""

import tomllib
from importlib.metadata import version
from pathlib import Path


async def test_home_and_version_match_installed_package_and_pyproject(client, pool, settings):
    response = await client.get("/api/v1/version")
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert response.status_code == 200
    assert response.json() == {
        "app": "fx-predictor-ai",
        "version": version("fx-predictor-ai"),
    }
    assert response.json()["version"] == metadata["project"]["version"]
    pool.fetchval.assert_not_awaited()

    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Hello, world!",
        "app": "fx-predictor-ai",
        "version": settings.version,
    }
