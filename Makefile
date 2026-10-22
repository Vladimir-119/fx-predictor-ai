UV = uv run --extra web --extra ml

# RustFS доступен клиентам через MLflow, без прямых S3-ссылок.
export MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD = false
export MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD = false
export MLFLOW_DISABLE_TELEMETRY = true

.PHONY: install hooks fmt lint test check db infra dataset eda train import-rates update-rates worker run up down logs psql inspect integration restart-api

install:
	uv sync --frozen --extra web --extra ml

hooks:
	$(UV) pre-commit install
	$(UV) pre-commit run --all-files

fmt:
	$(UV) ruff format .

lint:
	$(UV) ruff check .

test:
	$(UV) pytest -q -m "not integration"

check: fmt lint test

db:
	docker compose up -d --wait postgres
	docker compose run --rm db-init

infra:
	docker compose up -d --build --wait --wait-timeout 240 mlflow

dataset:
	$(UV) python -m src.ml.data

eda:
	$(UV) jupyter nbconvert --execute --to notebook --output eda.executed.ipynb --output-dir reports notebooks/eda.ipynb

train:
	$(UV) python -m src.ml.train

import-rates:
	$(UV) python -m src.services.rates import

update-rates:
	$(UV) python -m src.services.rates update

worker:
	docker compose --profile worker up -d --build worker

run:
	$(UV) uvicorn src.app:app --reload --no-access-log

up:
	docker compose up -d --build --wait --wait-timeout 240

restart-api:
	docker compose restart app

down:
	docker compose --profile worker --profile tasks down

logs:
	docker compose logs -f app

psql:
	docker compose exec postgres sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

inspect:
	docker image ls fx-predictor-ai:local
	docker compose exec app id -u

integration: export API_BASE_URL ?= http://127.0.0.1:8000
integration: export EXPECT_POSTGRES ?= healthy
integration:
	$(UV) pytest -q -m integration --no-cov

.PHONY: tasks tasks-two tasks-stop broker-test produce benchmark

tasks: db
	docker compose --profile tasks up -d --build --wait --wait-timeout 240 app inference-worker

tasks-two:
	docker compose --profile tasks up -d --scale inference-worker=2 --wait inference-worker

tasks-stop:
	docker compose --profile tasks stop inference-worker

broker-test: export RUN_BROKER_TESTS = 1
broker-test:
	$(UV) pytest -q tests/test_broker_integration.py --no-cov

produce:
	$(UV) python scripts/produce_tasks.py

benchmark:
	$(UV) python scripts/benchmark.py

.PHONY: task-file
task-file:
	$(UV) python scripts/produce_tasks.py --prepare
