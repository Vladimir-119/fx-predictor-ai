"""Настраиваемые эксперименты на одном снимке; отдельное продвижение модели."""

import argparse
import asyncio
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from matplotlib import pyplot as plt
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import Settings, get_settings
from src.ml.dataset import TrainingSnapshot, collect_snapshot, load_snapshot
from src.ml.features import FEATURE_COLUMNS, FEATURE_SET, make_training_frame
from src.ml.models import Candidate, ExperimentConfig, Regressor, create_model, read_config


def training_window(frame: pd.DataFrame, candidate: Candidate, minimum: int) -> pd.DataFrame:
    """Обрезать только обучение; даты validation и holdout остаются общими."""
    if candidate.train_years is not None:
        start = frame.target_date.max() - pd.DateOffset(years=candidate.train_years)
        frame = frame.loc[frame.target_date > start]
    if len(frame) < minimum:
        raise ValueError(f"{candidate.name}: at least {minimum} training rows are required")
    return frame


def fit_model(candidate: Candidate, frame: pd.DataFrame) -> Regressor:
    """Модель предсказывает изменение; признаки общие с inference."""
    model = create_model(candidate)
    model.fit(frame[FEATURE_COLUMNS], frame["target_change"])
    return model


def predict_rates(model: Regressor, frame: pd.DataFrame) -> np.ndarray:
    """Перейти от изменения к курсу в EUR за один USD."""
    return frame.rate.to_numpy() + np.asarray(model.predict(frame[FEATURE_COLUMNS])).reshape(-1)


def calculate_metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, float]:
    """Skill > 0 означает улучшение MAE относительно последнего известного курса."""
    actual = frame.target_rate.to_numpy()
    naive = float(mean_absolute_error(actual, frame.rate.to_numpy()))
    mae = float(mean_absolute_error(actual, predicted))
    metrics = {
        "mae": mae,
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "naive_mae": naive,
        "direction_accuracy": float(
            np.mean(
                np.sign(predicted - frame.rate.to_numpy())
                == np.sign(frame.target_change.to_numpy())
            )
        ),
    }
    # При неизменном ряде деление на ноль не имеет содержательного смысла.
    if naive > 0:
        metrics["skill_vs_naive"] = 1.0 - mae / naive
    return metrics


def split_frame(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Последний holdout не участвует в выборе параметров или победителя."""
    required = config.min_training_size + config.validation_size + config.holdout_size
    if len(frame) < required:
        raise ValueError(f"At least {required} examples are required for configured windows")
    development = frame.iloc[: -config.holdout_size]
    training = development.iloc[: -config.validation_size]
    return training, development.iloc[-config.validation_size :], frame.iloc[-config.holdout_size :]


def backtest(
    candidate: Candidate, development: pd.DataFrame, config: ExperimentConfig
) -> pd.DataFrame:
    """Общие expanding folds; train_years при необходимости делает окно скользящим."""
    folds = TimeSeriesSplit(
        n_splits=config.cv_splits, test_size=config.validation_size // config.cv_splits
    )
    records = []
    for fold, (train_indices, check_indices) in enumerate(folds.split(development), start=1):
        training = training_window(
            development.iloc[train_indices], candidate, config.min_training_size
        )
        checking = development.iloc[check_indices]
        started = time.perf_counter()
        model = fit_model(candidate, training)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predictions = predict_rates(model, checking)
        records.append(
            {
                "fold": fold,
                "train_rows": len(training),
                "train_target_start": str(training.target_date.min().date()),
                "train_target_end": str(training.target_date.max().date()),
                "validation_start": str(checking.target_date.min().date()),
                "validation_end": str(checking.target_date.max().date()),
                "fit_seconds": fit_seconds,
                "predict_ms_per_row": (time.perf_counter() - started) * 1000 / len(checking),
                **calculate_metrics(checking, predictions),
            }
        )
    return pd.DataFrame(records)


def log_input(frame: pd.DataFrame, context: str, snapshot: TrainingSnapshot) -> None:
    """Dataset inputs ссылаются на точный снимок; копия снимка лежит в артефактах."""
    dataset = mlflow.data.from_pandas(
        frame,
        source=snapshot.path.resolve().as_uri(),
        name=f"ECB USD/EUR {context}",
        targets="target_change",
    )
    mlflow.log_input(dataset, context=context)


def log_diagnostics(frame: pd.DataFrame, predictions: np.ndarray, label: str) -> None:
    """Сохранить прогнозы, временной график и распределение ошибок."""
    result = frame[["date", "target_date", "rate", "target_rate"]].copy()
    result["predicted_rate"] = predictions
    result["error"] = predictions - result.target_rate
    mlflow.log_text(result.to_csv(index=False), f"diagnostics/{label}_predictions.csv")
    figure, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(result.target_date, result.target_rate, label="Actual", linewidth=0.8)
    axes[0].plot(result.target_date, result.predicted_rate, label="Predicted", linewidth=0.8)
    axes[0].set(title=label, ylabel="EUR per USD")
    axes[0].legend()
    axes[1].hist(result.error, bins=50)
    axes[1].set(xlabel="Prediction error, EUR per USD", ylabel="Count")
    figure.tight_layout()
    mlflow.log_figure(figure, f"diagnostics/{label}.png")
    plt.close(figure)


def log_provenance(
    snapshot: TrainingSnapshot, config: ExperimentConfig, frame: pd.DataFrame
) -> None:
    """Версии данных, кода, конфигурации и признаков привязаны к каждому Run."""
    revision = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or "unknown"
    )
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        ).stdout.strip()
    )
    mlflow.log_params(
        {
            "dataset_version": snapshot.manifest["dataset_version"],
            "dataset_sha256": snapshot.manifest["sha256"],
            "pair": "USD/EUR",
            "feature_set": config.feature_set,
            "features": json.dumps(FEATURE_COLUMNS),
            "git_commit": revision,
            "git_dirty": dirty,
            "train_start": str(frame.target_date.min().date()),
            "train_end": str(frame.target_date.max().date()),
            "train_rows": len(frame),
            "target": "next_rate_change",
        }
    )
    mlflow.log_dict(snapshot.manifest, "data/manifest.json")
    mlflow.log_artifact(str(snapshot.path), artifact_path="data")
    mlflow.log_dict(config.model_dump(mode="json"), "config/experiments.json")
    for filename in ("features.py", "models.py", "train.py", "dataset.py"):
        mlflow.log_artifact(str(Path(__file__).with_name(filename)), artifact_path="code")
    for filename in ("uv.lock", "pyproject.toml"):
        mlflow.log_artifact(filename, artifact_path="environment")


def log_parameters(candidate: Candidate, model: Regressor) -> None:
    """Логировать фактические параметры, включая defaults фабрики."""
    mlflow.log_params(
        {
            "candidate": candidate.name,
            "model_type": candidate.model,
            "train_years": candidate.train_years or "all",
        }
    )
    mlflow.log_params(
        {
            key: value
            for key, value in model.get_params().items()
            if isinstance(value, (str, int, float, bool))
        }
    )


def register_model(
    settings: Settings, model: Regressor, frame: pd.DataFrame, snapshot: TrainingSnapshot
) -> str:
    """Контракт rates-v1 совместим со старым API; новые признаки потребуют новой версии."""
    example = frame[FEATURE_COLUMNS].head(3)
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
            for package in ("mlflow-skinny", "numpy", "pandas", "scikit-learn", "catboost")
        ],
    )
    registered = mlflow.register_model(info.model_uri, settings.model_name)
    client = MlflowClient()
    for key, value in {
        "pair": "USD/EUR",
        "dataset_sha256": snapshot.manifest["sha256"],
        "feature_set": FEATURE_SET,
    }.items():
        client.set_model_version_tag(settings.model_name, registered.version, key, str(value))
    return str(registered.version)


def train(settings: Settings, snapshot: TrainingSnapshot, config: ExperimentConfig) -> dict:
    """Выбрать по CV, один раз оценить holdout; champion никогда не меняется здесь."""
    frame = make_training_frame(snapshot.rates)
    training, validation, holdout = split_frame(frame, config)
    development = pd.concat([training, validation], ignore_index=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    scores = []
    with mlflow.start_run(run_name="training-batch", tags={"stage": "batch"}) as batch:
        mlflow.log_dict(config.model_dump(mode="json"), "config/experiments.json")
        mlflow.log_dict(snapshot.manifest, "data/manifest.json")
        for candidate in config.candidates:
            selected = training_window(training, candidate, config.min_training_size)
            with mlflow.start_run(
                run_name=candidate.name, nested=True, tags={"stage": "comparison"}
            ):
                model = fit_model(candidate, selected)
                log_provenance(snapshot, config, selected)
                log_parameters(candidate, model)
                log_input(selected, "training", snapshot)
                log_input(validation, "validation", snapshot)
                folds = backtest(candidate, development, config)
                metrics = {
                    f"cv_{key}": float(folds[key].mean())
                    for key in (
                        "mae",
                        "rmse",
                        "direction_accuracy",
                        "fit_seconds",
                        "predict_ms_per_row",
                    )
                }
                metrics["cv_mae_std"] = float(folds.mae.std(ddof=0))
                metrics["cv_naive_mae"] = float(folds.naive_mae.mean())
                if metrics["cv_naive_mae"] > 0:
                    metrics["cv_skill_vs_naive"] = 1 - metrics["cv_mae"] / metrics["cv_naive_mae"]
                mlflow.log_metrics(metrics)
                mlflow.log_text(folds.to_csv(index=False), "diagnostics/backtest.csv")
                predictions = predict_rates(model, validation)
                mlflow.log_metrics(
                    {
                        f"validation_{key}": value
                        for key, value in calculate_metrics(validation, predictions).items()
                    }
                )
                log_diagnostics(validation, predictions, "validation")
                if isinstance(model, CatBoostRegressor):
                    importance = pd.DataFrame(
                        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
                    )
                    mlflow.log_text(
                        importance.to_csv(index=False), "diagnostics/feature_importance.csv"
                    )
                registered = register_model(settings, model, selected, snapshot)
                scores.append({"candidate": candidate.name, "version": registered, **metrics})
        comparison = pd.DataFrame(scores).sort_values("cv_mae", kind="stable")
        mlflow.log_text(comparison.to_csv(index=False), "comparison.csv")
        winner = next(
            candidate
            for candidate in config.candidates
            if candidate.name == comparison.iloc[0].candidate
        )
        with mlflow.start_run(
            run_name=f"holdout-{winner.name}", nested=True, tags={"stage": "holdout"}
        ) as run:
            selected = training_window(development, winner, config.min_training_size)
            evaluated = fit_model(winner, selected)
            log_provenance(snapshot, config, selected)
            log_parameters(winner, evaluated)
            log_input(selected, "training", snapshot)
            log_input(holdout, "test", snapshot)
            predictions = predict_rates(evaluated, holdout)
            metrics = calculate_metrics(holdout, predictions)
            mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
            log_diagnostics(holdout, predictions, "holdout")
            evaluation_run = run.info.run_id
        with mlflow.start_run(
            run_name=f"candidate-{winner.name}", nested=True, tags={"stage": "candidate"}
        ):
            selected = training_window(frame, winner, config.min_training_size)
            production = fit_model(winner, selected)
            log_provenance(snapshot, config, selected)
            log_parameters(winner, production)
            log_input(selected, "training", snapshot)
            mlflow.log_param("evaluation_run_id", evaluation_run)
            candidate_version = register_model(settings, production, selected, snapshot)
        MlflowClient().set_registered_model_alias(
            settings.model_name, "candidate", candidate_version
        )
        result = {
            "winner": winner.name,
            "candidate_version": candidate_version,
            "batch_run_id": batch.info.run_id,
            "evaluation_run_id": evaluation_run,
            "snapshot_path": str(snapshot.path),
            "dataset_version": snapshot.manifest["dataset_version"],
            **{f"test_{key}": value for key, value in metrics.items()},
        }
        mlflow.log_dict(result, "result.json")
    return result


def promote(settings: Settings, selected_version: str) -> None:
    """Явное переключение после проверки совместимости; сервис перечитает при рестарте."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    client.get_model_version(settings.model_name, selected_version)
    predictor = mlflow.pyfunc.load_model(f"models:/{settings.model_name}/{selected_version}")
    metadata = predictor.metadata.metadata or {}
    if (
        metadata.get("pair") != "USD/EUR"
        or metadata.get("target") != "next_rate_change"
        or metadata.get("features") != FEATURE_COLUMNS
        or metadata.get("feature_set", FEATURE_SET) != FEATURE_SET
    ):
        raise ValueError("Model has an incompatible feature contract")
    client.set_registered_model_alias(settings.model_name, "champion", selected_version)


def main() -> None:
    """Без --snapshot обновить данные; --promote не запускает обучение."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.toml"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--promote", type=int)
    args = parser.parse_args()
    settings = get_settings()
    if args.promote is not None:
        if args.promote < 1:
            parser.error("--promote must be a positive model version")
        promote(settings, str(args.promote))
        print(f"champion -> {args.promote}; restart API and inference-worker")
        return
    config = read_config(args.config)
    snapshot = (
        load_snapshot(args.snapshot) if args.snapshot else asyncio.run(collect_snapshot(settings))
    )
    print(f"Training snapshot: {snapshot.path}", flush=True)
    result = train(settings, snapshot, config)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/training.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
