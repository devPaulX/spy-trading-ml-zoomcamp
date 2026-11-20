"""Train and evaluate chronological SPY return prediction models."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import DUMMIES, NUMERIC, TO_PREDICT


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "features.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODEL_OUTPUT = MODELS / "random_forest.joblib"
METADATA_OUTPUT = MODELS / "random_forest_metadata.json"
PREDICTIONS_OUTPUT = REPORTS / "test_predictions.csv"
METRICS_OUTPUT = REPORTS / "model_metrics.json"
IMPORTANCE_OUTPUT = REPORTS / "feature_importance.csv"
PLOT_OUTPUT = REPORTS / "model_evaluation.png"

FEATURE_COLUMNS = NUMERIC + DUMMIES
REQUIRED_COLUMNS = ["date", TO_PREDICT] + FEATURE_COLUMNS
TARGET_HORIZON = 5


def load_dataset(input_path: Path = INPUT) -> pd.DataFrame:
    """Load labeled feature rows sorted by date, validating required columns."""
    try:
        data = pd.read_csv(input_path, parse_dates=["date"])
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Unable to read feature data from {input_path}: {exc}") from exc

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Feature data is missing required columns: {missing_columns}")
    if data["date"].isna().any():
        raise ValueError("Feature data contains invalid or missing dates")

    data = data.dropna(subset=[TO_PREDICT]).copy()
    if data.empty:
        raise ValueError("Feature data contains no rows with observed targets")

    try:
        data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="raise"
        )
        data[TO_PREDICT] = pd.to_numeric(data[TO_PREDICT], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Targets and model features must be numeric: {exc}") from exc

    return data.sort_values("date", kind="mergesort").reset_index(drop=True)


def chronological_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically and purge each pre-test horizon boundary."""
    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)
    if train_end == 0 or validation_end <= train_end or validation_end >= len(data):
        raise ValueError("Not enough labeled rows for non-empty 70/15/15 splits")

    full_train = data.iloc[:train_end]
    full_validation = data.iloc[train_end:validation_end]
    if len(full_train) <= TARGET_HORIZON or len(full_validation) <= TARGET_HORIZON:
        raise ValueError("Not enough rows to purge the target horizon from each split")

    train = full_train.iloc[:-TARGET_HORIZON].copy()
    validation = full_validation.iloc[:-TARGET_HORIZON].copy()
    test = data.iloc[validation_end:].copy()

    if not (train["date"].max() < validation["date"].min()):
        raise ValueError("Training and validation date ranges overlap")
    if not (validation["date"].max() < test["date"].min()):
        raise ValueError("Validation and test date ranges overlap")
    if set(train["date"]).intersection(test["date"]) or set(
        validation["date"]
    ).intersection(test["date"]):
        raise ValueError("Training or validation dates overlap test dates")
    if train.index.max() + TARGET_HORIZON >= validation.index.min():
        raise ValueError("A training target window crosses into validation")
    if validation.index.max() + TARGET_HORIZON >= test.index.min():
        raise ValueError("A validation target window crosses into test")

    return train, validation, test


def validate_model_inputs(*partitions: pd.DataFrame) -> None:
    """Ensure feature inputs and targets are finite before model fitting."""
    if TO_PREDICT in FEATURE_COLUMNS:
        raise ValueError("The target must not be included in model input features")
    for partition in partitions:
        values = partition[FEATURE_COLUMNS].to_numpy(dtype=float)
        targets = partition[TO_PREDICT].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Model inputs contain NaN or infinite values")
        if not np.isfinite(targets).all():
            raise ValueError("Target values contain NaN or infinite values")


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Calculate regression and direction metrics with safe undefined handling."""
    if not np.isfinite(predicted).all():
        raise ValueError("Predictions contain NaN or infinite values")

    correlation = pd.Series(actual).corr(pd.Series(predicted), method="pearson")
    r2 = r2_score(actual, predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2) if np.isfinite(r2) else None,
        "prediction_actual_pearson_correlation": (
            float(correlation) if pd.notna(correlation) else None
        ),
        "directional_accuracy": float(
            np.mean(np.sign(actual) == np.sign(predicted))
        ),
    }


def date_boundaries(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """Return inclusive date boundaries for each chronological split."""
    return {
        name: {
            "start": partition["date"].min().date().isoformat(),
            "end": partition["date"].max().date().isoformat(),
        }
        for name, partition in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }


def purged_date_boundaries(data: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return date boundaries for the rows removed at both split boundaries."""
    train_end = int(len(data) * 0.70)
    validation_end = int(len(data) * 0.85)
    purged_train = data.iloc[train_end - TARGET_HORIZON : train_end]
    purged_validation = data.iloc[
        validation_end - TARGET_HORIZON : validation_end
    ]
    return {
        "after_train": {
            "start": purged_train["date"].min().date().isoformat(),
            "end": purged_train["date"].max().date().isoformat(),
        },
        "after_validation": {
            "start": purged_validation["date"].min().date().isoformat(),
            "end": purged_validation["date"].max().date().isoformat(),
        },
    }


def save_evaluation_plot(
    test: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    feature_importance: pd.DataFrame,
) -> None:
    """Save chronological actual/predicted returns and feature importance plots."""
    figure, (returns_axis, importance_axis) = plt.subplots(
        nrows=1, ncols=2, figsize=(15, 6)
    )
    returns_axis.plot(test["date"], actual, label="Actual", linewidth=1)
    returns_axis.plot(test["date"], predicted, label="Predicted", linewidth=1)
    returns_axis.axhline(0, color="black", linewidth=0.8)
    returns_axis.set_title("Test-set actual versus predicted returns")
    returns_axis.set_xlabel("Date")
    returns_axis.set_ylabel("Future 5-day return")
    returns_axis.legend()
    returns_axis.tick_params(axis="x", rotation=30)

    importance = feature_importance.sort_values("importance", ascending=True)
    importance.plot(
        kind="barh",
        x="feature",
        y="importance",
        legend=False,
        ax=importance_axis,
        color="#4c72b0",
    )
    importance_axis.set_title("Random Forest feature importance")
    importance_axis.set_xlabel("Impurity-based importance")
    importance_axis.set_ylabel("Feature")

    figure.tight_layout()
    figure.savefig(PLOT_OUTPUT, dpi=150)
    plt.close(figure)


def train_and_evaluate(input_path: Path = INPUT) -> dict[str, Any]:
    """Fit the baseline and Random Forest, compare validation, and evaluate test."""
    if not FEATURE_COLUMNS or TO_PREDICT in FEATURE_COLUMNS:
        raise ValueError("Model features must be defined separately from the target")

    data = load_dataset(input_path)
    train, validation, test = chronological_split(data)
    validate_model_inputs(train, validation, test)

    x_train = train[FEATURE_COLUMNS]
    x_validation = validation[FEATURE_COLUMNS]
    x_test = test[FEATURE_COLUMNS]
    y_train = train[TO_PREDICT].to_numpy(dtype=float)
    y_validation = validation[TO_PREDICT].to_numpy(dtype=float)
    y_test = test[TO_PREDICT].to_numpy(dtype=float)

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(x_train, y_train)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    validation_metrics = {
        "baseline": calculate_metrics(
            y_validation, baseline.predict(x_validation)
        ),
        "random_forest": calculate_metrics(
            y_validation, model.predict(x_validation)
        ),
    }
    baseline_test_predictions = baseline.predict(x_test)
    test_predictions = model.predict(x_test)
    test_metrics = {
        "baseline": calculate_metrics(y_test, baseline_test_predictions),
        "random_forest": calculate_metrics(y_test, test_predictions),
    }
    if len(test_predictions) != len(test):
        raise ValueError("Test prediction count does not match the test split")
    if not np.isfinite(test_predictions).all() or not np.isfinite(
        baseline_test_predictions
    ).all():
        raise ValueError("Test predictions contain NaN or infinite values")

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_OUTPUT)

    boundaries = date_boundaries(train, validation, test)
    metadata = {
        "target": TO_PREDICT,
        "target_horizon_trading_days": TARGET_HORIZON,
        "features": FEATURE_COLUMNS,
        "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "purged_rows": {
            "after_train_before_validation": TARGET_HORIZON,
            "after_validation_before_test": TARGET_HORIZON,
        },
        "purged_date_boundaries": purged_date_boundaries(data),
        "split_date_boundaries": boundaries,
        "random_forest_parameters": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
        "directional_accuracy": (
            "Fraction of rows where sign(prediction) equals sign(actual); "
            "zero is treated as a neutral direction."
        ),
    }
    METADATA_OUTPUT.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    predictions = pd.DataFrame(
        {
            "date": test["date"].dt.strftime("%Y-%m-%d"),
            "actual_return": y_test,
            "predicted_return": test_predictions,
            "baseline_prediction": baseline_test_predictions,
        }
    )
    predictions.to_csv(PREDICTIONS_OUTPUT, index=False)

    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_OUTPUT, index=False)

    metrics = {
        "target": TO_PREDICT,
        "target_horizon_trading_days": TARGET_HORIZON,
        "feature_names": FEATURE_COLUMNS,
        "split_rows": metadata["split_rows"],
        "purged_rows": metadata["purged_rows"],
        "split_date_boundaries": boundaries,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    METRICS_OUTPUT.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_evaluation_plot(test, y_test, test_predictions, importance)

    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "baseline_mae": test_metrics["baseline"]["mae"],
        "model_mae": test_metrics["random_forest"]["mae"],
        "model_rmse": test_metrics["random_forest"]["rmse"],
        "model_r2": test_metrics["random_forest"]["r2"],
        "directional_accuracy": test_metrics["random_forest"][
            "directional_accuracy"
        ],
    }


def main() -> None:
    """Train models, persist artifacts, and print test metrics."""
    try:
        results = train_and_evaluate()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Model training failed: {exc}") from exc

    print(f"TRAIN ROWS: {results['train_rows']}")
    print(f"VALIDATION ROWS: {results['validation_rows']}")
    print(f"TEST ROWS: {results['test_rows']}")
    print(f"BASELINE MAE: {results['baseline_mae']:.8f}")
    print(f"MODEL MAE: {results['model_mae']:.8f}")
    print(f"MODEL RMSE: {results['model_rmse']:.8f}")
    r2 = results["model_r2"]
    print(f"MODEL R2: {r2:.8f}" if r2 is not None else "MODEL R2: undefined")
    print(f"DIRECTIONAL ACCURACY: {results['directional_accuracy']:.8f}")
    print("MODEL TRAINING COMPLETE")


if __name__ == "__main__":
    main()
