"""Build leakage-safe features for the SPY five-day return model."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "market_data.csv"
OUTPUT = ROOT / "data" / "features.csv"

TO_PREDICT = "future_5d_return"

NUMERIC = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ma_5_ratio",
    "ma_20_ratio",
    "volatility_20d",
    "volume_ratio_20d",
    "yield_lag_1d",
    "yield_change_5d",
]
DUMMIES = [
    "weekday_Monday",
    "weekday_Tuesday",
    "weekday_Wednesday",
    "weekday_Thursday",
    "weekday_Friday",
]

FEATURE_COLUMNS = NUMERIC + DUMMIES
REQUIRED_COLUMNS = ["date", "spy_close", "spy_volume", "us10y_yield"]


def build_features(input_path: Path = INPUT) -> pd.DataFrame:
    """Load market data and return chronologically ordered, leakage-safe features."""
    try:
        data = pd.read_csv(input_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to read market data from {input_path}: {exc}") from exc

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Market data is missing required columns: {missing_columns}")
    if data.empty:
        raise ValueError(f"Market data contains no records: {input_path}")

    data["date"] = pd.to_datetime(data["date"], errors="raise")
    for column in ("spy_close", "spy_volume", "us10y_yield"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Market data contains missing dates or feature source values")
    if (data["spy_close"] <= 0).any():
        raise ValueError("SPY close prices must be positive")

    data = data.sort_values("date", kind="mergesort").reset_index(drop=True)
    close = data["spy_close"]

    data[TO_PREDICT] = close.shift(-5) / close - 1
    data["ret_1d"] = close.pct_change(fill_method=None)
    data["ret_5d"] = close.pct_change(periods=5, fill_method=None)
    data["ret_20d"] = close.pct_change(periods=20, fill_method=None)
    data["ma_5_ratio"] = close / close.rolling(window=5, min_periods=5).mean() - 1
    data["ma_20_ratio"] = close / close.rolling(window=20, min_periods=20).mean() - 1
    data["volatility_20d"] = data["ret_1d"].rolling(
        window=20, min_periods=20
    ).std()
    data["volume_ratio_20d"] = data["spy_volume"] / data["spy_volume"].rolling(
        window=20, min_periods=20
    ).mean()
    data["yield_lag_1d"] = data["us10y_yield"].shift(1)
    data["yield_change_5d"] = data["yield_lag_1d"].diff(periods=5)

    weekday = data["date"].dt.day_of_week
    for day, feature in enumerate(DUMMIES):
        data[feature] = weekday.eq(day).astype("int8")

    data = data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return data


def main() -> None:
    """Build and save the feature dataset, reporting its basic dimensions."""
    try:
        data = build_features()
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(OUTPUT, index=False)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Feature generation failed: {exc}") from exc

    print(f"ROW COUNT: {len(data)}")
    print(f"DATE RANGE: {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"FEATURE COUNT: {len(FEATURE_COLUMNS)}")
    print(f"MISSING TARGET COUNT: {data[TO_PREDICT].isna().sum()}")
    print(f"SAVED: {OUTPUT}")


if __name__ == "__main__":
    main()
