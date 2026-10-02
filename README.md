# fx-predictor-ai

Авторы проекта:

- Volkov Vladimir — Telegram [@The_Valdemare](https://t.me/The_Valdemare), [thevaldemare1@gmail.com](mailto:thevaldemare1@gmail.com).
- Vagin Arseniy — Telegram [@quitepeaky](https://t.me/quitepeaky), [arseniyjvagin@gmail.com](mailto:arseniyjvagin@gmail.com).

Проект для прогнозирования валютных курсов. Текущая версия содержит асинхронный API на FastAPI, подключение к PostgreSQL, проверки состояния сервисов, JSON-логи и CI/CD. Модель прогнозирования пока не реализована.

Используем Python 3.12, uv, FastAPI, asyncpg, PostgreSQL 16, Ruff, pytest, pre-commit и Docker Compose.

## Быстрый запуск

Понадобятся Git, uv, Docker с Compose и Make. На macOS запустите Docker Desktop. Все команды выполняются из корня проекта.

Склонируйте репозиторий, заменив `YOUR_LOGIN` на имя владельца:

```bash
git clone https://github.com/YOUR_LOGIN/fx-predictor-ai.git
cd fx-predictor-ai
cp .env.example .env
```

В `.env` замените пароль в **двух местах**: `POSTGRES_PASSWORD` и `DATABASE_URL`. Для локальной разработки удобно использовать пароль из латинских букв и цифр. `.env` не добавляем в Git.

```bash
make install
make check
make up
make integration
```

Откройте [Swagger UI](http://localhost:8000/docs). В нём можно отправлять запросы через **Try it out → Execute**. Описание API также доступно в [OpenAPI JSON](http://localhost:8000/openapi.json).

При успешном запуске контейнеры `app` и `postgres` имеют статус `healthy`, тесты проходят, а `/api/v1/health` возвращает HTTP `200` и `status: ok`.

PostgreSQL работает в Docker: отдельно устанавливать его на компьютер не нужно. Данные сохраняются в томе `postgres-data`.

## Локальная разработка

API можно запускать на компьютере с автоматическим перезапуском, оставив PostgreSQL в контейнере:

```bash
make install
make db
make run
```

Если весь стек уже запущен через Compose, сначала выполните `make down`, чтобы освободить порт `8000`. Для остановки локального API нажмите `Ctrl+C`.

`make run` использует `DATABASE_URL` из `.env`, обычно с адресом `localhost:5433`. В контейнере приложение подключается к `postgres:5432`; Compose собирает этот URL автоматически.

`APP_PORT` меняет порт приложения в Compose, но не порт локального `make run`. Локальный запуск на другом порту:

```bash
uv run --extra web uvicorn src.app:app --reload --port 8001 --no-access-log
```

## API

| Метод и путь | Назначение |
|---|---|
| `GET /healthz` | Быстрая проверка процесса приложения без обращения к БД |
| `GET /api/v1/version` | Название и версия установленного приложения |
| `GET /api/v1/health` | Состояние приложения и PostgreSQL, версии и время проверки |
| `GET /` | Приветственное сообщение, название и версия приложения |

Проверка из терминала:

```bash
curl -i http://localhost:8000/healthz
curl -i http://localhost:8000/api/v1/version
curl -i http://localhost:8000/api/v1/health
```

`/healthz` возвращает `{"status": "ok"}`, даже если база недоступна. Путь не зависит от версии прикладного API, поэтому не содержит `/api/v1`.

Пример ответа `/api/v1/version`:

```json
{"app": "fx-predictor-ai", "version": "0.1.0"}
```

Версия задаётся в `[project].version` в `pyproject.toml` и читается из метаданных установленного пакета. `v1` в пути — версия API, а `0.1.0` — версия приложения.

`/api/v1/health` выполняет SQL-запрос к PostgreSQL и возвращает:

- `status`: `ok` при успехе или `degraded` при недоступности БД;
- `environment`: название окружения;
- `dependencies`: состояние, версия и `response_time_ms` каждого проверяемого компонента.

Время измеряется в миллисекундах. Для PostgreSQL оно включает получение соединения из пула, запрос и возврат соединения. При ошибке подключения или таймауте ответ имеет HTTP `503`, статус БД `unavailable` и её версию `null`. Сообщение об успешном старте API само по себе не подтверждает подключение к БД: первое соединение создаётся при проверке.

## Настройки и зависимости

`.env.example` хранится в репозитории как шаблон. `.env` содержит локальные настройки. Приложение читает их через `Settings` в `src/config.py`; переменные окружения имеют приоритет над `.env`.

| Переменная | Значение по умолчанию в шаблоне | Назначение |
|---|---|---|
| `POSTGRES_USER` | `fx_predictor` | Пользователь БД |
| `POSTGRES_PASSWORD` | `change-me-before-use` | Пароль; замените в своём `.env` |
| `POSTGRES_DB` | `fx_predictor_ai` | Имя базы |
| `POSTGRES_PORT` | `5433` | Порт БД на компьютере |
| `APP_PORT` | `8000` | Порт API при запуске через Compose |
| `DATABASE_URL` | См. `.env.example` | Подключение к БД для локального API |
| `LOG_LEVEL` | `INFO` | Уровень подробности логов |
| `ENVIRONMENT` | `development` | Название окружения в health-ответе |
| `HEALTH_TIMEOUT_SECONDS` | `3` | Таймаут проверки БД, в секундах |
| `DB_POOL_MAX_SIZE` | `5` | Максимум соединений в пуле |
| `APP_IMAGE` | `fx-predictor-ai:local` | Необязательный выбор готового образа для Compose |

Если меняете пользователя, пароль, имя базы или `POSTGRES_PORT`, обновите соответствующие части `DATABASE_URL`. Порты Compose доступны только с локального компьютера.

Зависимости описаны в `pyproject.toml`, версии зафиксированы в `uv.lock`:

- `web` — FastAPI, Uvicorn, pydantic-settings и asyncpg;
- `dev` — Ruff, pre-commit, pytest и инструменты тестирования.

`make install` выполняет `uv sync --extra web`: устанавливает приложение, `web` и стандартную группу `dev`. Активировать `.venv` вручную не требуется. В Docker и CI используется `--frozen` для установки по lock-файлу; в образ приложения dev-зависимости не включаются.

## Проверки и тесты

```bash
make fmt          # Форматирование с изменением файлов
make lint         # Проверка кода
make test         # Изолированные тесты и покрытие
make check        # fmt, lint и test по очереди
```

Ruff настроен в `pyproject.toml`: Python 3.12, длина строки 100 символов, проверка импортов, именования, аннотаций и распространённых ошибок. Для проверки форматирования без изменения файлов:

```bash
uv run --extra web ruff format --check .
```

В проекте 18 изолированных и 2 интеграционных теста. Изолированные тесты подменяют БД, поэтому `make test` работает без PostgreSQL. Они проверяют ответы API, ошибки, таймауты, логи и работу пула.

Покрытие измеряется по строкам и ветвлениям. Порог проекта — **95%**. Отчёты: сводка в терминале, `htmlcov/index.html` для браузера и `coverage.xml` для CI.

Для проверки работающего API с настоящей БД:

```bash
make up
make integration
```

Интеграционные тесты ожидают API на порту `8000`. Если в Compose выбран другой порт, например `8001`:

```bash
API_BASE_URL=http://127.0.0.1:8001 EXPECT_POSTGRES=healthy uv run --extra web pytest -q -m integration --no-cov
```

### Отказ БД и восстановление

После запуска стека остановите PostgreSQL:

```bash
docker compose stop postgres
curl -i http://localhost:8000/healthz
curl -i http://localhost:8000/api/v1/health
```

Ожидается HTTP `200` от `/healthz` и HTTP `503` со статусом `degraded` от `/api/v1/health`. Автоматическая проверка этого состояния:

```bash
API_BASE_URL=http://127.0.0.1:8000 EXPECT_POSTGRES=unavailable uv run --extra web pytest -q -m integration --no-cov
```

Восстановите БД и повторите тесты:

```bash
make db
make integration
```

API должен восстановить работу с базой без перезапуска приложения.

## Контейнеры и логи

```bash
docker compose ps    # Состояние контейнеров
make logs            # Логи API; Ctrl+C завершает просмотр
make psql            # SQL-консоль; выход через \q
make inspect         # Размер образа и UID пользователя приложения
make down            # Остановка контейнеров с сохранением данных БД
```

Dockerfile использует два этапа: сборку и запуск. В итоговый образ переносим готовое окружение и uv; API запускается от пользователя `appuser` с UID `10001`. Зависимости устанавливаются отдельным слоем для кеширования. Compose задаёт сеть, том БД, проверки здоровья, ограничения ресурсов и ротацию логов.

Логи выводятся в stdout в формате JSON. Для HTTP-запросов записываются метод, путь, статус, длительность и `request_id`. Идентификатор также возвращается в заголовке `X-Request-ID`:

```bash
curl -i http://localhost:8000/api/v1/health
docker compose logs --tail=20 app
```

По этому идентификатору можно найти соответствующую запись `request_completed`. Настройки подключения не включаются в обычные сообщения об ошибках БД.

PostgreSQL создаёт пользователя и базу при первом запуске с пустым томом. Изменение пароля в `.env` не меняет пароль уже созданного пользователя. Для полного сброса локальной базы **с удалением данных**:

```bash
docker compose down -v
make up
```

## Участие в разработке

Установите проверки перед коммитом после клонирования:

```bash
make install
make hooks
```

pre-commit проверяет код, форматирование, пробелы, окончания файлов и синтаксис YAML/TOML/JSON. Если подходящих файлов нет, проверка показывает `Skipped`. Новые файлы нужно добавить через `git add`, чтобы они попали в общую проверку. Если хук исправил файлы, просмотрите изменения и добавьте их повторно.

Для изменений создавайте отдельную ветку, например `feat/currency-prediction` или `fix/health-check`, и открывайте Pull Request в основную ветку:

```bash
git switch -c feat/currency-prediction
# Внесите изменения.
make check
git diff
git add .
git diff --cached --name-only
git commit -m "feat: add currency prediction"
git push -u origin HEAD
```

Не добавляйте `.env`, `.venv`, кеши и отчёты покрытия. Перед слиянием дождитесь успешных проверок в GitHub. Предпочитаем сообщения с понятным типом изменения: `feat:`, `fix:`, `docs:`, `test:` или `ci:`.

## CI/CD и релизы

Workflow находится в `.github/workflows/ci-cd.yml`:

1. `quality` проверяет зависимости, pre-commit, Ruff, тесты, покрытие и версию релизного тега.
2. `integration` проверяет Docker-образ, доступность PostgreSQL, отказ и восстановление БД.
3. `publish` после успешных проверок публикует образ в GitHub Container Registry (GHCR).
4. `release` после публикации релизного образа создаёт страницу версии в **Releases** с описанием изменений и командой скачивания образа.

| Событие | Проверки | Публикация образа | GitHub Release |
|---|---|---|---|
| Push в `main` или `master` | Да | `edge` и `sha-<SHA коммита>` | Нет |
| Push в рабочую ветку или Pull Request | Да | Нет | Нет |
| Push тега `vMAJOR.MINOR.PATCH` | Да | Версия релиза и `sha-<SHA коммита>` | Да |

Проверяем все ветки, чтобы находить ошибки до слияния. `edge` содержит актуальную сборку основной ветки, а релизный тег обозначает конкретную версию. При ошибке проверок образ и Release не публикуются. При ошибке публикации образа Release тоже не создаётся.

Для GHCR используется встроенный `GITHUB_TOKEN`. После push проверьте вкладку **Actions**, отчёт `coverage` и итог задачи `publish`: имя образа, теги и digest. Образ появляется в разделе **Packages** владельца репозитория.

Релизный Git-тег должен совпадать с версией в `pyproject.toml`. Например, для версии `0.1.0`:

```bash
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

После отправки тега дождитесь успешного завершения **Actions**. В **Releases** автоматически появится `v0.1.0`, а в **Packages** — Docker-образ с тегом `0.1.0`. Вручную создавать страницу релиза не нужно. При повторном запуске workflow существующий Release сохраняется.

Номер версии выбирает разработчик. Для следующего релиза, например `0.1.1`, измените `version` в `pyproject.toml`, выполните `uv lock`, `make install` и `make check`, затем закоммитьте и отправьте изменения:

```bash
git add pyproject.toml uv.lock
git commit -m "chore: bump version to 0.1.1"
git push
git tag -a v0.1.1 -m "Release 0.1.1"
git push origin v0.1.1
```

Опубликованные релизные теги не перезаписываем.

Запуск опубликованного образа, когда релиз уже появился в GHCR:

```bash
docker pull ghcr.io/YOUR_LOGIN/fx-predictor-ai:0.1.0
APP_IMAGE=ghcr.io/YOUR_LOGIN/fx-predictor-ai:0.1.0 docker compose up -d --no-build --wait --wait-timeout 120
curl -i http://localhost:8000/api/v1/version
curl -i http://localhost:8000/api/v1/health
```

Замените `YOUR_LOGIN` на владельца пакета в нижнем регистре. Для приватного пакета потребуется `docker login ghcr.io`. Для запуска точного содержимого образа можно использовать digest из итогов публикации вместо тега. Вернуться к локальной сборке можно через `make up`, если `APP_IMAGE` не задан постоянно в окружении или `.env`.

## Структура проекта

| Путь | Назначение |
|---|---|
| `src/app.py` | Создание приложения, жизненный цикл и логи запросов |
| `src/api/` | Эндпоинты API |
| `src/services/health.py` | Проверки компонентов и сбор отчёта |
| `src/schemas.py` | Схемы ответов |
| `src/config.py` | Настройки и версия приложения |
| `src/db.py` | Асинхронный пул PostgreSQL |
| `src/logging_config.py` | JSON-логи и контекст запроса |
| `tests/` | Изолированные и интеграционные тесты |
| `pyproject.toml`, `uv.lock` | Зависимости и настройки инструментов |
| `Dockerfile`, `docker-compose.yml` | Сборка и запуск контейнеров |
| `Makefile` | Команды разработчика |

## Частые проблемы

| Проблема | Решение |
|---|---|
| Docker не отвечает | Запустите Docker Desktop и проверьте `docker info` |
| Порт `8000` занят | Остановите второй экземпляр API; не запускайте `make run` и `make up` на одном порту |
| `/healthz` отвечает, а `/api/v1/health` возвращает `503` | Проверьте состояние БД, `DATABASE_URL`, пароль и логи PostgreSQL |
| Локальный API возвращает `500` при проверке БД | Проверьте формат `DATABASE_URL` и ошибку в логах; после изменения `.env` перезапустите `make run` |
| После смены пароля нет подключения | Существующий том сохраняет прежний пароль пользователя PostgreSQL |
| pre-commit исправил файлы | Просмотрите изменения, выполните `git add` и повторите проверку |
| Интеграционные тесты пропущены | Запустите API и БД, затем `make integration` |
| Версия приложения не обновилась | Выполните `make install` и перезапустите API либо пересоберите контейнер |
| `publish` пропущен | Проверьте событие запуска, ветку и результаты предыдущих задач |
