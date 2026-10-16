"""Признаки используют только известные значения; цель — следующий курс."""

import pandas as pd

FEATURE_COLUMNS = [
    "rate",
    "lag_1",
    "lag_2",
    "lag_5",
    "change_1",
    "mean_5",
    "mean_20",
    "std_20",
    "day_of_week",
]
MIN_HISTORY = 20


def build_features(rates: pd.DataFrame) -> pd.DataFrame:
    """Рассчитать признаки на конец каждой даты без будущих значений."""
    if rates["date"].duplicated().any() or not rates["date"].is_monotonic_increasing:
        raise ValueError("History must be sorted and dates must be unique")
    frame = rates.copy()
    for lag in (1, 2, 5):
        frame[f"lag_{lag}"] = frame["rate"].shift(lag)
    frame["change_1"] = frame["rate"].diff()
    frame["mean_5"] = frame["rate"].rolling(5).mean()
    frame["mean_20"] = frame["rate"].rolling(20).mean()
    frame["std_20"] = frame["rate"].rolling(20).std()
    frame["day_of_week"] = frame["date"].dt.dayofweek
    return frame


def make_training_frame(rates: pd.DataFrame) -> pd.DataFrame:
    """Связать признаки даты t с изменением курса на следующую публикацию."""
    frame = build_features(rates)
    frame["target_date"] = frame["date"].shift(-1)
    frame["target_rate"] = frame["rate"].shift(-1)
    frame["target_change"] = frame["target_rate"] - frame["rate"]
    return frame.dropna().reset_index(drop=True)


def latest_features(rates: pd.DataFrame) -> pd.DataFrame:
    """Вернуть одну строку признаков для уже известной истории."""
    if len(rates) < MIN_HISTORY:
        raise ValueError(f"At least {MIN_HISTORY} observations are required")
    return build_features(rates).iloc[[-1]][FEATURE_COLUMNS]


def split_by_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Отделить 70% обучения, 15% validation и 15% последнего holdout."""
    if len(frame) < 100:
        raise ValueError("At least 100 training examples are required")
    train_end = int(len(frame) * 0.70)
    validation_end = int(len(frame) * 0.85)
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )
