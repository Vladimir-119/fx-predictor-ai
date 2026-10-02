.PHONY: install hooks lint fmt test check run up down logs psql integration

install:
	uv sync --extra web

hooks:
	uv run --extra web pre-commit install
	uv run --extra web pre-commit run --all-files

fmt:
	uv run --extra web ruff format .

lint:
	uv run --extra web ruff check .

test:
	uv run --extra web pytest -q -m "not integration"

check: fmt lint test

db:
	docker compose up -d --wait postgres

run:
	uv run --extra web uvicorn src.app:app --reload --no-access-log

up:
	docker compose up -d --build --wait --wait-timeout 120

down:
	docker compose down

logs:
	docker compose logs -f app

psql:
	docker compose exec postgres sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

inspect:
	docker image ls fx-predictor-ai:local
	docker compose exec app id -u

integration:
	API_BASE_URL=http://127.0.0.1:8000 EXPECT_POSTGRES=healthy uv run --frozen --extra web pytest -q -m integration --no-cov
