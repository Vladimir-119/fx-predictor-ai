#!/bin/sh
set -eu
export PGPASSWORD="$POSTGRES_PASSWORD"
psql -h postgres -U "$POSTGRES_USER" -d postgres -v ON_ERROR_STOP=1 -v mlflow_db="$MLFLOW_DB" <<'SQL'
SELECT format('CREATE DATABASE %I', :'mlflow_db')
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'mlflow_db')
\gexec
SQL
psql -h postgres -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -f /sql/exchange_rates.sql
