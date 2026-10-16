"""Настройки читаются из env и валидируются до запуска."""

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings


def test_settings_read_environment_validate_values_and_hide_password(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:private@localhost:5433/db")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("HEALTH_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "7")
    settings = get_settings()
    assert settings.log_level == "DEBUG"
    assert settings.health_timeout_seconds == 2.5
    assert settings.db_pool_max_size == 7
    assert "private" not in repr(settings)
    assert "private" not in settings.model_dump_json()

    with pytest.raises(ValidationError):
        Settings(_env_file=None, health_timeout_seconds=0)
