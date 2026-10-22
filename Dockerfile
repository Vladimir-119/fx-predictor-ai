# Сборка: создаём окружение с зависимостями и приложением.
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# Зависимости устанавливаем отдельно, чтобы кешировать их при изменении кода.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra web --no-install-project

# Устанавливаем приложение в окружение без привязки к исходным файлам.
COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev --extra web --no-editable

# Запуск: переносим uv и готовое окружение в итоговый образ.
FROM python:3.12-slim AS runtime
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false \
    MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false \
    MLFLOW_DISABLE_TELEMETRY=true
LABEL org.opencontainers.image.title="fx-predictor-ai"

# Приложение работает от отдельного пользователя без прав root.
RUN useradd --uid 10001 --create-home appuser
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
USER appuser

# Порт API и проверка доступности приложения.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD uv run --no-project python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

# Запускаем API через uv.
CMD ["uv", "run", "--no-project", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
