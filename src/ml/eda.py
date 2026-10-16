"""Краткий EDA выбранного ряда и артефакты в MLflow."""

import mlflow
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from src.config import Settings
from src.ml.data import RAW_PATH, read_manifest


def make_plots(rates: pd.DataFrame) -> dict[str, Figure]:
    """Показать историю, изменения, их распределение и изменчивость."""
    changes = rates["rate"].diff()
    figures = {}
    series = {
        "rate": (rates["rate"], "USD/EUR: EUR per 1 USD"),
        "changes": (changes, "Change between successive publications"),
        "volatility": (changes.rolling(20).std(), "20-observation change volatility"),
    }
    for name, (values, title) in series.items():
        figure, axis = plt.subplots(figsize=(10, 3))
        axis.plot(rates["date"], values, linewidth=0.8)
        axis.set(title=title, xlabel="Publication date", ylabel="EUR per USD")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figures[name] = figure
    figure, axis = plt.subplots(figsize=(7, 3))
    axis.hist(changes.dropna(), bins=80)
    axis.set(title="Distribution of daily changes", xlabel="Change, EUR per USD", ylabel="Count")
    figure.tight_layout()
    figures["change_distribution"] = figure
    return figures


def make_report(rates: pd.DataFrame) -> str:
    """Сформировать воспроизводимые выводы и ограничения анализа."""
    changes = rates["rate"].diff()
    largest = rates.loc[changes.abs().nlargest(5).index].copy()
    largest["change"] = changes.loc[largest.index]
    return (
        "# ECB USD/EUR: data review\n\n"
        f"Observations: {len(rates)}. Dates: {rates.date.min().date()} — "
        f"{rates.date.max().date()}. Missing values: {rates.isna().sum().sum()}. "
        f"Duplicate dates: {rates.date.duplicated().sum()}.\n\n"
        "The source quotes EUR/USD. We take its reciprocal: EUR per one USD. "
        "Only reference rates are used. Weekends and holidays are not filled with copies.\n\n"
        f"Minimum: {rates.rate.min():.6f}; maximum: {rates.rate.max():.6f}. "
        f"Standard deviation of one-step changes: {changes.std():.6f}.\n\n"
        "Largest absolute changes (retained; a large move is not automatically an error):\n\n"
        f"```text\n{largest.to_string(index=False)}\n```\n\n"
        "The level and volatility change over time, so we use chronological validation. "
        "The target is the next publication, not a weekend market quote. "
        "Training predicts a change; the API adds it to the last known rate. "
        "The last-value baseline is essential. News and causal effects are not studied here. "
        "This EDA checks data quality; model selection uses training/validation only.\n"
    )


def log_eda(settings: Settings, rates: pd.DataFrame, figures: dict[str, Figure]) -> str:
    """Сохранить Dataset, исходный файл, графики и выводы в один Run."""
    manifest = read_manifest()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name="EDA", tags={"stage": "eda", "pair": "USD/EUR"}) as run:
        dataset = mlflow.data.from_pandas(
            rates, source=str(manifest["source"]), name="ECB USD/EUR snapshot"
        )
        mlflow.log_input(dataset, context="eda")
        mlflow.log_params(
            {"dataset_version": manifest["dataset_version"], "sha256": manifest["sha256"]}
        )
        mlflow.log_metrics({"observations": len(rates), "missing_values": rates.isna().sum().sum()})
        mlflow.log_dict(manifest, "data/manifest.json")
        mlflow.log_artifact(str(RAW_PATH), artifact_path="data")
        mlflow.log_text(make_report(rates), "eda/report.md")
        for name, figure in figures.items():
            mlflow.log_figure(figure, f"eda/{name}.png")
        return run.info.run_id
