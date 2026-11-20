"""Создать настоящий persistence baseline для пустого стенда, сохранив существующий alias."""

import logging
from importlib.metadata import version

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.dummy import DummyRegressor

from src.config import Settings, get_settings
from src.logging_config import setup_logging
from src.ml.data import RAW_PATH, load_rates, read_manifest
from src.ml.features import FEATURE_COLUMNS, FEATURE_SET, make_training_frame

log = logging.getLogger(__name__)


def selected_model(client: MlflowClient, settings: Settings) -> ModelVersion | None:
    """Отсутствие alias допускает bootstrap; ошибка подключения не означает пустой Registry."""
    try:
        registered = client.get_registered_model(settings.model_name)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None
    if settings.model_alias not in registered.aliases:
        return None
    return client.get_model_version_by_alias(settings.model_name, settings.model_alias)


def bootstrap(settings: Settings) -> str:
    """Это стартовая модель стенда, а не участник сравнения качества или замена champion."""
    if not settings.model_alias:
        raise ValueError("MODEL_ALIAS must be set for the observability bootstrap")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    existing = selected_model(client, settings)
    if existing is not None:
        log.info(
            "bootstrap_existing_model_kept",
            extra={"model_alias": settings.model_alias, "model_version": str(existing.version)},
        )
        return str(existing.version)
    # load_rates проверяет SHA-256 закреплённого локального снимка.
    frame = make_training_frame(load_rates())
    if frame.empty:
        raise ValueError("Bootstrap dataset must contain at least 21 observations")
    model = DummyRegressor(strategy="constant", constant=0)
    model.fit(frame[FEATURE_COLUMNS], frame["target_change"])
    example = frame[FEATURE_COLUMNS].head(3)
    mlflow.set_experiment(f"{settings.mlflow_experiment}-observability-bootstrap")
    with mlflow.start_run(
        run_name="persistence-baseline",
        tags={"purpose": "stand-bootstrap", "model_type": "persistence-naive"},
    ):
        mlflow.log_params(
            {"strategy": "constant", "constant_change": 0, "training_rows": len(frame)}
        )
        dataset = mlflow.data.from_pandas(
            frame,
            source=RAW_PATH.resolve().as_uri(),
            name="ECB USD/EUR observability bootstrap",
            targets="target_change",
        )
        mlflow.log_input(dataset, context="training")
        mlflow.log_dict(read_manifest(), "dataset-manifest.json")
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=example,
            signature=infer_signature(example, model.predict(example)),
            metadata={
                "pair": "USD/EUR",
                "target": "next_rate_change",
                "features": FEATURE_COLUMNS,
                "feature_set": FEATURE_SET,
            },
            pip_requirements=[
                f"{package}=={version(package)}"
                for package in ("mlflow-skinny", "numpy", "pandas", "scikit-learn")
            ],
        )
        registered = mlflow.register_model(info.model_uri, settings.model_name)
    # Повторно проверяем alias: появившийся за время регистрации champion сохраняется.
    existing = selected_model(client, settings)
    selected_version = str(existing.version if existing is not None else registered.version)
    if existing is None:
        client.set_registered_model_alias(
            settings.model_name, settings.model_alias, selected_version
        )
    log.info(
        "bootstrap_model_ready",
        extra={"model_alias": settings.model_alias, "model_version": selected_version},
    )
    return selected_version


def main() -> None:
    """Одноразовый Compose init: ошибка должна остановить запуск воркера, а не скрыться."""
    settings = get_settings()
    setup_logging(settings.log_level)
    try:
        bootstrap(settings)
    except Exception:
        log.exception("bootstrap_model_failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
