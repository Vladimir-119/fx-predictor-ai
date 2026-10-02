"""Настройки из переменных окружения; версия из метаданных пакета."""

from importlib.metadata import version as package_version
from typing import ClassVar, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Собрать настройки из env или локального .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: ClassVar[str] = "fx-predictor-ai"
    version: ClassVar[str] = package_version(app_name)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    environment: str = "development"
    database_url: SecretStr = SecretStr("postgresql://localhost:5433/fx_predictor_ai")
    health_timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    db_pool_max_size: int = Field(default=5, ge=1, le=100)


def get_settings() -> Settings:
    """Вернуть настройки приложения."""
    return Settings()
