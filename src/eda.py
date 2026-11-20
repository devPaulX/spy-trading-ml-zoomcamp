"""Summarize the target and feature correlations for the SPY dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.features import DUMMIES, FEATURE_COLUMNS, NUMERIC, TO_PREDICT


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "features.csv"
REPORTS = ROOT / "reports"
SUMMARY_OUTPUT = REPORTS / "eda_summary.md"
CORRELATIONS_OUTPUT = REPORTS / "feature_correlations.csv"
PLOT_OUTPUT = REPORTS / "eda.png"

FEATURE_DEFINITIONS = {
    "ret_1d": "SPY close percentage return over the previous trading day.",
    "ret_5d": "SPY close percentage return over the previous five trading days.",
    "ret_20d": "SPY close percentage return over the previous twenty trading days.",
    "ma_5_ratio": "Current SPY close divided by its trailing five-day average close, minus one.",
    "ma_20_ratio": "Current SPY close divided by its trailing twenty-day average close, minus one.",
    "volatility_20d": "Trailing twenty-day sample standard deviation of one-day SPY returns.",
    "volume_ratio_20d": "Current SPY volume divided by its trailing twenty-day average volume.",
    "yield_lag_1d": "US 10-year Treasury yield from the previous trading day.",
    "yield_change_5d": "Five-trading-day change in the one-day-lagged US 10-year yield.",
    "weekday_Monday": "Indicator that the observation date is a Monday.",
    "weekday_Tuesday": "Indicator that the observation date is a Tuesday.",
    "weekday_Wednesday": "Indicator that the observation date is a Wednesday.",
    "weekday_Thursday": "Indicator that the observation date is a Thursday.",
    "weekday_Friday": "Indicator that the observation date is a Friday.",
}


def load_features(input_path: Path = INPUT) -> pd.DataFrame:
    """Read the feature dataset and validate columns needed for EDA."""
    try:
        data = pd.read_csv(input_path, parse_dates=["date"])
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Unable to read feature data from {input_path}: {exc}") from exc

    required = ["date", TO_PREDICT] + FEATURE_COLUMNS
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Feature data is missing required columns: {missing_columns}")
    if data[TO_PREDICT].notna().sum() == 0:
        raise ValueError("Feature data has no non-missing target values to analyze")
    return data


def calculate_correlations(data: pd.DataFrame) -> pd.Series:
    """Calculate Pearson correlations between engineered features and the target."""
    observed = data.loc[data[TO_PREDICT].notna(), FEATURE_COLUMNS + [TO_PREDICT]]
    return observed[FEATURE_COLUMNS].corrwith(observed[TO_PREDICT], method="pearson")


def write_summary(data: pd.DataFrame, correlations: pd.Series) -> None:
    """Write target statistics, feature definitions, and correlations as Markdown."""
    target = data[TO_PREDICT].dropna()
    stats = target.describe()
    summary = [
        "# Exploratory Data Analysis",
        "",
        "## Target",
        "",
        f"`{TO_PREDICT}` is the SPY close price five trading days ahead divided by "
        "the current close, minus one. The final five observations have no known "
        "target yet and are excluded from target statistics and correlations.",
        "",
        "### Summary statistics",
        "",
        "| Statistic | Value |",
        "| --- | ---: |",
    ]
    summary.extend(f"| {name} | {value:.8f} |" for name, value in stats.items())
    summary.extend(
        [
            "",
            f"Observed targets: {len(target)}",
            f"Missing targets: {data[TO_PREDICT].isna().sum()}",
            "",
            "## Feature definitions",
            "",
            "All rolling windows use current and prior observations only. "
            "`yield_lag_1d` uses the previous trading day's yield.",
            "",
            "| Feature | Definition |",
            "| --- | --- |",
        ]
    )
    summary.extend(
        f"| `{feature}` | {FEATURE_DEFINITIONS[feature]} |"
        for feature in NUMERIC + DUMMIES
    )
    summary.extend(
        [
            "",
            "## Pearson correlations with target",
            "",
            "Correlations use rows with observed targets.",
            "",
            "| Feature | Correlation |",
            "| --- | ---: |",
        ]
    )
    summary.extend(
        f"| `{feature}` | {correlations[feature]:.8f} |"
        if pd.notna(correlations[feature])
        else f"| `{feature}` | N/A |"
        for feature in FEATURE_COLUMNS
    )
    SUMMARY_OUTPUT.write_text("\n".join(summary) + "\n", encoding="utf-8")


def write_plot(data: pd.DataFrame, correlations: pd.Series) -> None:
    """Save a target histogram and feature-correlation bar chart."""
    target = data[TO_PREDICT].dropna()
    ordered = correlations.sort_values()

    figure, (histogram_axis, correlation_axis) = plt.subplots(
        nrows=1, ncols=2, figsize=(15, 6)
    )
    histogram_axis.hist(target, bins=40, edgecolor="black", alpha=0.8)
    histogram_axis.set_title("Future 5-day SPY return")
    histogram_axis.set_xlabel(TO_PREDICT)
    histogram_axis.set_ylabel("Count")

    colors = ["#c44e52" if value < 0 else "#4c72b0" for value in ordered]
    ordered.plot(kind="barh", ax=correlation_axis, color=colors)
    correlation_axis.axvline(0, color="black", linewidth=0.8)
    correlation_axis.set_title("Feature correlation with target")
    correlation_axis.set_xlabel("Pearson correlation")
    correlation_axis.set_ylabel("Feature")

    figure.tight_layout()
    figure.savefig(PLOT_OUTPUT, dpi=150)
    plt.close(figure)


def main() -> None:
    """Generate EDA outputs and print a completion marker."""
    try:
        data = load_features()
        REPORTS.mkdir(parents=True, exist_ok=True)
        correlations = calculate_correlations(data)
        correlations.rename("pearson_correlation").rename_axis("feature").to_csv(
            CORRELATIONS_OUTPUT
        )
        write_summary(data, correlations)
        write_plot(data, correlations)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"EDA failed: {exc}") from exc

    print("EDA COMPLETE")


if __name__ == "__main__":
    main()
