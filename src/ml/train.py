"""Baseline, Ridge, CatBoost: временная проверка, holdout и Registry."""

import json
import subprocess
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
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Settings, get_settings
from src.ml.data import load_rates, read_manifest
from src.ml.features import FEATURE_COLUMNS, make_training_frame, split_by_time

Regressor = DummyRegressor | Pipeline | CatBoostRegressor


def create_model(name: str) -> Regressor:
    """Создать независимую модель с фиксированными параметрами."""
    if name == "baseline":
        return DummyRegressor(strategy="constant", constant=0.0)
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "catboost":
        return CatBoostRegressor(
            iterations=400,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=5,
            loss_function="RMSE",
            random_seed=42,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown model: {name}")


def fit_model(name: str, frame: pd.DataFrame) -> Regressor:
    """Обучить предсказание изменения, используя только переданную выборку."""
    model = create_model(name)
    model.fit(frame[FEATURE_COLUMNS], frame["target_change"])
    return model


def predict_rates(model: Regressor, frame: pd.DataFrame) -> np.ndarray:
    """Прибавить прогноз изменения к известному курсу."""
    return frame["rate"].to_numpy() + np.asarray(model.predict(frame[FEATURE_COLUMNS])).reshape(-1)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Измерить MAE и RMSE в EUR за один USD."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
    }


def backtest(name: str, development: pd.DataFrame, validation_size: int) -> pd.DataFrame:
    """Проверить три последовательных окна с расширяющимся обучением."""
    folds = TimeSeriesSplit(n_splits=3, test_size=validation_size // 3)
    records = []
    for fold, (train_indices, check_indices) in enumerate(folds.split(development), start=1):
        training = development.iloc[train_indices]
        checking = development.iloc[check_indices]
        model = fit_model(name, training)
        metrics = calculate_metrics(checking.target_rate.to_numpy(), predict_rates(model, checking))
        records.append(
            {
                "fold": fold,
                "train_target_end": training.target_date.max().isoformat(),
                "validation_start": checking.target_date.min().isoformat(),
                "validation_end": checking.target_date.max().isoformat(),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def log_input(frame: pd.DataFrame, context: str) -> None:
    """Связать конкретное окно данных с Run через Dataset Tracking."""
    dataset = mlflow.data.from_pandas(
        frame,
        source=str(read_manifest()["source"]),
        name=f"ECB USD/EUR {context}",
        targets="target_change",
    )
    mlflow.log_input(dataset, context=context)


def log_diagnostics(frame: pd.DataFrame, predictions: np.ndarray, label: str) -> None:
    """Сохранить прогнозы, временной график и распределение ошибок."""
    result = frame[["date", "target_date", "rate", "target_rate"]].copy()
    result["predicted_rate"] = predictions
    result["error"] = predictions - result["target_rate"]
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


def register_model(settings: Settings, model: Regressor, frame: pd.DataFrame) -> str:
    """Зарегистрировать совместимую версию с контрактом признаков."""
    manifest = read_manifest()
    example = frame[FEATURE_COLUMNS].head(3)
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        input_example=example,
        signature=infer_signature(example, model.predict(example)),
        metadata={"pair": "USD/EUR", "target": "next_rate_change", "features": FEATURE_COLUMNS},
        pip_requirements=[
            f"{package}=={version(package)}"
            for package in ("mlflow-skinny", "numpy", "pandas", "scikit-learn", "catboost")
        ],
    )
    registered = mlflow.register_model(info.model_uri, settings.model_name)
    client = MlflowClient()
    client.set_model_version_tag(settings.model_name, registered.version, "pair", "USD/EUR")
    client.set_model_version_tag(
        settings.model_name, registered.version, "dataset_sha256", str(manifest["sha256"])
    )
    return str(registered.version)


def log_provenance(frame: pd.DataFrame) -> None:
    """Сохранить снимок, преобразования, границы обучения и Git SHA."""
    manifest = read_manifest()
    revision = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or "unknown"
    )
    mlflow.log_params(
        {
            "dataset_version": manifest["dataset_version"],
            "dataset_sha256": manifest["sha256"],
            "pair": "USD/EUR",
            "features": json.dumps(FEATURE_COLUMNS),
            "git_commit": revision,
            "train_start": str(frame.target_date.min().date()),
            "train_end": str(frame.target_date.max().date()),
            "seed": 42,
            "target": "next publication change; predicted rate = known rate + change",
        }
    )
    mlflow.log_dict(manifest, "data/manifest.json")
    mlflow.log_artifact(str(Path(__file__).with_name("features.py")), artifact_path="code")
    mlflow.log_artifact(__file__, artifact_path="code")


def train(settings: Settings, frame: pd.DataFrame) -> dict[str, str | float]:
    """Сравнить кандидатов, оценить победителя и обучить рабочую версию."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    training, validation, holdout = split_by_time(frame)
    development = pd.concat([training, validation], ignore_index=True)
    scores = {}
    versions = {}
    for name in ("baseline", "ridge", "catboost"):
        with mlflow.start_run(run_name=name, tags={"stage": "comparison"}):
            model = fit_model(name, training)
            log_provenance(training)
            mlflow.log_params(
                {
                    key: value
                    for key, value in model.get_params().items()
                    if isinstance(value, (str, int, float))
                }
            )
            log_input(training, "training")
            log_input(validation, "validation")
            folds = backtest(name, development, len(validation))
            scores[name] = float(folds.mae.mean())
            mlflow.log_metrics({"cv_mae": scores[name], "cv_rmse": float(folds.rmse.mean())})
            mlflow.log_text(folds.to_csv(index=False), "diagnostics/backtest.csv")
            metrics = calculate_metrics(
                validation.target_rate.to_numpy(), predict_rates(model, validation)
            )
            mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
            log_diagnostics(validation, predict_rates(model, validation), "validation")
            if isinstance(model, CatBoostRegressor):
                importance = pd.DataFrame(
                    {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
                )
                mlflow.log_text(
                    importance.to_csv(index=False), "diagnostics/feature_importance.csv"
                )
            versions[name] = register_model(settings, model, training)
    winner = min(scores, key=scores.get)
    challenger = next(name for name in sorted(scores, key=scores.get) if name != winner)
    with mlflow.start_run(run_name=f"holdout-{winner}", tags={"stage": "holdout"}) as run:
        evaluated = fit_model(winner, development)
        log_provenance(development)
        log_input(development, "training")
        log_input(holdout, "test")
        predictions = predict_rates(evaluated, holdout)
        metrics = calculate_metrics(holdout.target_rate.to_numpy(), predictions)
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        log_diagnostics(holdout, predictions, "holdout")
        evaluation_run = run.info.run_id
    with mlflow.start_run(run_name=f"production-{winner}", tags={"stage": "production"}):
        production = fit_model(winner, frame)
        log_provenance(frame)
        log_input(frame, "training")
        mlflow.log_params({"selected_model": winner, "evaluation_run_id": evaluation_run})
        champion_version = register_model(settings, production, frame)
    client = MlflowClient()
    client.set_registered_model_alias(settings.model_name, "champion", champion_version)
    client.set_registered_model_alias(settings.model_name, "challenger", versions[challenger])
    return {
        "winner": winner,
        "champion_version": champion_version,
        "challenger_version": versions[challenger],
        "evaluation_run_id": evaluation_run,
        "test_mae": metrics["mae"],
        "test_rmse": metrics["rmse"],
    }


def main() -> None:
    """Запустить воспроизводимый цикл экспериментов."""
    result = train(get_settings(), make_training_frame(load_rates()))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/training.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
