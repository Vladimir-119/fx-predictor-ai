"""Параметры экспериментов и небольшая фабрика моделей."""

import tomllib
from pathlib import Path
from typing import Literal

from catboost import CatBoostRegressor
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.features import FEATURE_SET

Regressor = DummyRegressor | Pipeline | CatBoostRegressor


class Candidate(BaseModel):
    """Один Run; окно обучения ограничивается независимо от дат проверки."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    model: Literal["baseline", "ridge", "catboost"]
    train_years: int | None = Field(default=None, ge=1)
    params: dict[str, int | float | str | bool] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Размеры окон задаются в публикациях, а не в календарных днях."""

    model_config = ConfigDict(extra="forbid")
    feature_set: Literal["rates-v1"] = FEATURE_SET
    validation_size: int = Field(default=126, ge=3)
    holdout_size: int = Field(default=63, ge=1)
    cv_splits: int = Field(default=3, ge=2)
    min_training_size: int = Field(default=252, ge=20)
    candidates: list[Candidate] = Field(min_length=2)

    @model_validator(mode="after")
    def check_candidates(self) -> "ExperimentConfig":
        """Без naive baseline или с одинаковыми именами сравнение неоднозначно."""
        names = [candidate.name for candidate in self.candidates]
        baselines = [candidate for candidate in self.candidates if candidate.model == "baseline"]
        if len(names) != len(set(names)) or len(baselines) != 1:
            raise ValueError("Use unique candidate names and exactly one baseline")
        if baselines[0].params or baselines[0].train_years is not None:
            raise ValueError("The naive baseline must not have parameters or a training window")
        if self.validation_size < self.cv_splits:
            raise ValueError("validation_size must be at least cv_splits")
        return self


def read_config(path: Path) -> ExperimentConfig:
    """Неизвестные поля считаются ошибкой, а не игнорируются."""
    return ExperimentConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def create_model(candidate: Candidate) -> Regressor:
    """Общие defaults можно переопределить явно через TOML."""
    if candidate.model == "baseline":
        return DummyRegressor(strategy="constant", constant=0.0)
    if candidate.model == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**{"alpha": 10.0, **candidate.params}))
    return CatBoostRegressor(
        **{
            "iterations": 400,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5,
            "loss_function": "MAE",
            "random_seed": 42,
            "thread_count": 2,
            "verbose": False,
            "allow_writing_files": False,
            **candidate.params,
        }
    )
