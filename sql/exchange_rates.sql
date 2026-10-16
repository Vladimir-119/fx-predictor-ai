CREATE TABLE IF NOT EXISTS exchange_rates (
    source TEXT NOT NULL,
    base TEXT NOT NULL,
    quote TEXT NOT NULL,
    rate_type TEXT NOT NULL,
    rate_date DATE NOT NULL,
    value DOUBLE PRECISION NOT NULL CHECK (value > 0),
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, base, quote, rate_type, rate_date)
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source TEXT NOT NULL,
    pair TEXT NOT NULL,
    as_of DATE NOT NULL,
    forecast_date DATE NOT NULL,
    predicted_rate DOUBLE PRECISION NOT NULL CHECK (predicted_rate > 0),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
