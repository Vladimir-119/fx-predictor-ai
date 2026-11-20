UV = uv run --extra web --extra ml
EXPERIMENT_CONFIG ?= configs/experiments.toml

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
	$(UV) python -m src.ml.train --config "$(EXPERIMENT_CONFIG)"

.PHONY: training-data train-snapshot promote backfill-rates restart-model
training-data:
	$(UV) python -m src.ml.dataset

train-snapshot:
	$(UV) python -m src.ml.train --config "$(EXPERIMENT_CONFIG)" --snapshot "$(SNAPSHOT)"

promote:
	$(UV) python -m src.ml.train --promote "$(VERSION)"

backfill-rates:
	$(UV) python -m src.services.rates backfill

restart-model:
	docker compose --profile tasks restart app inference-worker

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
	docker compose --profile "*" down

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

# Наблюдаемость включается отдельно от повседневной разработки.
COUNT ?= 300
CONCURRENCY ?= 20
MODE ?= normal
DEMO_COUNT ?= $(if $(filter overload,$(MODE)),$(COUNT),1)

.PHONY: obs-up obs-down obs-source obs-demo obs-two obs-one obs-stats obs-logs obs-traces obs-metrics
obs-up obs-source obs-two obs-one: export OTEL_SDK_DISABLED = false

obs-up:
	docker compose --profile observability up -d --build --wait --wait-timeout 360

obs-down:
	docker compose --profile observability stop alloy loki tempo otel-collector prometheus grafana

obs-source:
	$(UV) python scripts/produce_tasks.py --prepare --count $(COUNT)
	docker compose --profile observability run --rm source --file /app/reports/tasks.jsonl --output /app/reports/producer.csv --count $(COUNT)

obs-demo:
	$(UV) python scripts/observability_demo.py --mode $(MODE) --count $(DEMO_COUNT) --concurrency $(CONCURRENCY)

obs-two:
	docker compose --profile observability up -d --scale inference-worker=2 --wait inference-worker

obs-one:
	docker compose --profile observability up -d --scale inference-worker=1 --wait inference-worker

obs-stats:
	docker compose --profile observability stats --no-stream

obs-logs:
	docker compose --profile logs up -d --build --wait loki alloy

obs-traces:
	docker compose --profile traces up -d --build --wait tempo otel-collector

obs-metrics:
	docker compose --profile metrics up -d --build --wait prometheus otel-collector

.PHONY: obs-test
obs-test: export RUN_OBSERVABILITY_TESTS = 1
obs-test:
	$(UV) python -c "from dotenv import load_dotenv; load_dotenv(); import pytest; raise SystemExit(pytest.main(['-q', 'tests/test_observability_integration.py', '--no-cov']))"
