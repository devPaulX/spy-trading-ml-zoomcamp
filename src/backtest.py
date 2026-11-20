"""Backtest the SPY model's out-of-sample predictions at executable prices."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_INPUT = ROOT / "reports" / "test_predictions.csv"
MARKET_DATA_INPUT = ROOT / "data" / "market_data.csv"
MODEL_METRICS_INPUT = ROOT / "reports" / "model_metrics.json"
REPORTS = ROOT / "reports"
RESULTS_OUTPUT = REPORTS / "backtest_results.csv"
METRICS_OUTPUT = REPORTS / "backtest_metrics.json"
PLOT_OUTPUT = REPORTS / "backtest_equity.png"
SUMMARY_OUTPUT = REPORTS / "backtest_summary.md"

INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST_RATE = 0.001
TRADING_DAYS_PER_YEAR = 252
TARGET_COLUMN = "future_5d_return"


def load_test_predictions(
    predictions_path: Path = PREDICTIONS_INPUT,
    metrics_path: Path = MODEL_METRICS_INPUT,
) -> pd.DataFrame:
    """Load predictions within the recorded holdout test period only.

    The actual target column is deliberately not loaded or used by the strategy.
    """
    try:
        predictions = pd.read_csv(
            predictions_path,
            usecols=["date", "predicted_return"],
            parse_dates=["date"],
        )
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, pd.errors.ParserError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load test predictions or model metrics: {exc}") from exc

    if predictions.empty:
        raise ValueError("Test prediction file contains no predictions")
    if predictions["date"].isna().any():
        raise ValueError("Test predictions contain invalid or missing signal dates")
    if not predictions["date"].is_monotonic_increasing:
        raise ValueError("Signal dates must be strictly increasing")
    if predictions["date"].duplicated().any():
        raise ValueError("Signal dates must be unique")
    predictions["predicted_return"] = pd.to_numeric(
        predictions["predicted_return"], errors="coerce"
    )
    if not np.isfinite(predictions["predicted_return"].to_numpy(dtype=float)).all():
        raise ValueError("Predicted returns contain missing or infinite values")

    try:
        split_dates = metadata["split_date_boundaries"]
        test_rows = int(metadata["split_rows"]["test"])
        train_end = pd.Timestamp(split_dates["train"]["end"])
        validation_end = pd.Timestamp(split_dates["validation"]["end"])
        test_start = pd.Timestamp(split_dates["test"]["start"])
        test_end = pd.Timestamp(split_dates["test"]["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Model metrics must include split row counts and train, validation, "
            "and test date boundaries"
        ) from exc

    signal_dates = predictions["date"]
    if not train_end < validation_end < test_start <= test_end:
        raise ValueError("Recorded train, validation, and test periods are not chronological")
    if (
        len(predictions) != test_rows
        or signal_dates.min() != test_start
        or signal_dates.max() != test_end
    ):
        raise ValueError("Predictions do not match the recorded out-of-sample test split")
    if not validation_end < signal_dates.min():
        raise ValueError("Test predictions overlap training or model-tuning data")

    return predictions


def load_market_data(market_path: Path = MARKET_DATA_INPUT) -> pd.DataFrame:
    """Load and validate dates and open prices needed for execution."""
    try:
        market = pd.read_csv(market_path, usecols=["date", "spy_open"], parse_dates=["date"])
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Unable to read market open prices from {market_path}: {exc}") from exc

    if market.empty or market["date"].isna().any():
        raise ValueError("Market data must contain valid dates and at least one row")
    market["spy_open"] = pd.to_numeric(market["spy_open"], errors="coerce")
    if market["date"].duplicated().any():
        raise ValueError("Market data dates must be unique")
    market = market.sort_values("date", kind="mergesort").reset_index(drop=True)
    if not market["date"].is_monotonic_increasing:
        raise ValueError("Market dates must be strictly increasing")
    return market


def build_backtest(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """Apply signals to the next open-to-open interval, including trading costs."""
    if TARGET_COLUMN in predictions.columns:
        raise ValueError("Forward target values must not enter trading decisions")

    market_index = market.set_index("date")
    market_dates = market["date"].tolist()
    date_positions = {date: index for index, date in enumerate(market_dates)}
    executable_rows: list[dict[str, Any]] = []

    for prediction in predictions.itertuples(index=False):
        signal_date = prediction.date
        signal_position = date_positions.get(signal_date)
        if signal_position is None:
            continue
        start_position = signal_position + 1
        end_position = signal_position + 2
        if end_position >= len(market_dates):
            continue

        execution_start_date = market_dates[start_position]
        execution_end_date = market_dates[end_position]
        if not signal_date < execution_start_date < execution_end_date:
            raise ValueError("Execution dates must follow the signal date in order")
        start_open = market_index.at[execution_start_date, "spy_open"]
        end_open = market_index.at[execution_end_date, "spy_open"]
        if not np.isfinite(start_open) or not np.isfinite(end_open):
            continue
        if start_open <= 0 or end_open <= 0:
            raise ValueError("Execution open prices must be positive")

        # Only the prediction generated at the signal-date close determines exposure.
        predicted_return = float(prediction.predicted_return)
        position = int(predicted_return > 0)
        executable_rows.append(
            {
                "signal_date": signal_date,
                "predicted_return": predicted_return,
                "position": position,
                "execution_start_date": execution_start_date,
                "execution_end_date": execution_end_date,
                "spy_open_start": float(start_open),
                "spy_open_end": float(end_open),
                "benchmark_gross_return": float(end_open / start_open - 1),
            }
        )

    if not executable_rows:
        raise ValueError("No test predictions have the required future execution prices")

    results = pd.DataFrame(executable_rows)
    results["benchmark_execution_start_date"] = results["execution_start_date"]
    results["benchmark_execution_end_date"] = results["execution_end_date"]
    if not results["signal_date"].is_monotonic_increasing:
        raise ValueError("Executable signal dates must remain strictly increasing")
    if results["signal_date"].duplicated().any():
        raise ValueError("Executable signal dates must remain unique")
    if not (results["execution_start_date"] > results["signal_date"]).all():
        raise ValueError("A trade would execute on or before its signal date")
    if not (
        results["execution_end_date"] > results["execution_start_date"]
    ).all():
        raise ValueError("Execution intervals must be strictly increasing")

    results["gross_return"] = (
        results["position"] * results["benchmark_gross_return"]
    )
    results["position_change"] = results["position"].diff().abs()
    results.loc[0, "position_change"] = abs(results.loc[0, "position"])
    results["position_change"] = results["position_change"].astype(int)
    results["final_liquidation"] = False
    results.loc[results.index[-1], "final_liquidation"] = True

    strategy_equity = INITIAL_CAPITAL
    benchmark_equity = INITIAL_CAPITAL
    strategy_transaction_costs: list[float] = []
    strategy_cost_returns: list[float] = []
    strategy_net_returns: list[float] = []
    strategy_equities: list[float] = []
    benchmark_transaction_costs: list[float] = []
    benchmark_net_returns: list[float] = []
    benchmark_equities: list[float] = []
    benchmark_position = 1
    previous_benchmark_position = 0

    for row in results.itertuples():
        strategy_start_equity = strategy_equity
        strategy_turnover = int(row.position_change)
        strategy_entry_cost = (
            strategy_start_equity * TRANSACTION_COST_RATE * strategy_turnover
        )
        strategy_before_exit = (
            strategy_start_equity - strategy_entry_cost
        ) * (1 + row.gross_return)
        liquidation_turnover = int(row.final_liquidation and row.position == 1)
        strategy_exit_cost = (
            strategy_before_exit * TRANSACTION_COST_RATE * liquidation_turnover
        )
        strategy_equity = strategy_before_exit - strategy_exit_cost
        strategy_cost = strategy_entry_cost + strategy_exit_cost
        strategy_transaction_costs.append(strategy_cost)
        strategy_cost_returns.append(strategy_cost / strategy_start_equity)
        strategy_net_returns.append(strategy_equity / strategy_start_equity - 1)
        strategy_equities.append(strategy_equity)

        benchmark_start_equity = benchmark_equity
        benchmark_turnover = abs(benchmark_position - previous_benchmark_position)
        benchmark_entry_cost = (
            benchmark_start_equity * TRANSACTION_COST_RATE * benchmark_turnover
        )
        benchmark_before_exit = (
            benchmark_start_equity - benchmark_entry_cost
        ) * (1 + row.benchmark_gross_return)
        benchmark_exit_cost = (
            benchmark_before_exit
            * TRANSACTION_COST_RATE
            * int(row.final_liquidation)
        )
        benchmark_equity = benchmark_before_exit - benchmark_exit_cost
        benchmark_transaction_costs.append(
            benchmark_entry_cost + benchmark_exit_cost
        )
        benchmark_net_returns.append(
            benchmark_equity / benchmark_start_equity - 1
        )
        benchmark_equities.append(benchmark_equity)
        previous_benchmark_position = benchmark_position

    results["transaction_cost"] = strategy_transaction_costs
    results["transaction_cost_return"] = strategy_cost_returns
    results["net_return"] = strategy_net_returns
    results["strategy_equity"] = strategy_equities
    results["benchmark_transaction_cost"] = benchmark_transaction_costs
    results["benchmark_net_return"] = benchmark_net_returns
    results["benchmark_equity"] = benchmark_equities
    results["final_liquidation_cost"] = 0.0
    if results.loc[results.index[-1], "position"] == 1:
        results.loc[results.index[-1], "final_liquidation_cost"] = (
            results.loc[results.index[-1], "strategy_equity"]
            * TRANSACTION_COST_RATE
            / (1 - TRANSACTION_COST_RATE)
        )

    validate_backtest(predictions, results)
    return results


def calculate_metrics(
    returns: pd.Series,
    equity: pd.Series,
    transaction_costs: pd.Series,
    position_changes: int,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict[str, Any]:
    """Calculate annualized performance and trading-cost metrics."""
    values = returns.to_numpy(dtype=float)
    equity_values = equity.to_numpy(dtype=float)
    total_return = float(equity_values[-1] / initial_capital - 1)
    annualized_return = float((equity_values[-1] / initial_capital) ** (
        TRADING_DAYS_PER_YEAR / len(values)
    ) - 1)
    annualized_volatility = float(np.std(values, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(values) > 1 else 0.0
    daily_volatility = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sharpe_ratio = (
        float(np.mean(values) / daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR))
        if daily_volatility > 0
        else None
    )
    curve = np.concatenate(([initial_capital], equity_values))
    drawdown = curve / np.maximum.accumulate(curve) - 1
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "maximum_drawdown": float(drawdown.min()),
        "position_changes": int(position_changes),
        "transaction_costs": float(transaction_costs.sum()),
    }


def validate_backtest(predictions: pd.DataFrame, results: pd.DataFrame) -> None:
    """Assert timing, exposure, target isolation, and accounting invariants."""
    if TARGET_COLUMN in predictions.columns or TARGET_COLUMN in results.columns:
        raise AssertionError("Forward target values must not be used by the backtest")
    if not predictions["date"].is_monotonic_increasing or predictions["date"].duplicated().any():
        raise AssertionError("Signal dates must be strictly increasing and unique")
    if not results["signal_date"].is_monotonic_increasing or results["signal_date"].duplicated().any():
        raise AssertionError("Executed signal dates must be strictly increasing and unique")
    if not (results["execution_start_date"] > results["signal_date"]).all():
        raise AssertionError("A trade did not execute strictly after its signal")
    if not (
        results["execution_end_date"] > results["execution_start_date"]
    ).all():
        raise AssertionError("An execution interval is not chronological")
    if not (
        (results["position"] == (results["predicted_return"] > 0).astype(int)).all()
    ):
        raise AssertionError("Positions do not match the prediction-only decision rule")
    if not results["position"].isin([0, 1]).all():
        raise AssertionError("Exposure must always be zero or one")

    return_columns = [
        "benchmark_gross_return",
        "gross_return",
        "transaction_cost_return",
        "net_return",
        "benchmark_net_return",
        "transaction_cost",
        "benchmark_transaction_cost",
        "final_liquidation_cost",
    ]
    return_values = results[return_columns].to_numpy(dtype=float)
    if not np.isfinite(return_values).all():
        raise AssertionError("Returns must not be missing or infinite")
    if (results[["transaction_cost", "benchmark_transaction_cost"]] < 0).any().any():
        raise AssertionError("Transaction costs must be nonnegative")
    if not results["execution_start_date"].equals(
        results["benchmark_execution_start_date"]
    ) or not results["execution_end_date"].equals(
        results["benchmark_execution_end_date"]
    ):
        raise AssertionError("Strategy and benchmark return intervals do not align")
    expected_benchmark_returns = (
        results["spy_open_end"].to_numpy(dtype=float)
        / results["spy_open_start"].to_numpy(dtype=float)
        - 1
    )
    if not np.allclose(
        results["benchmark_gross_return"].to_numpy(), expected_benchmark_returns
    ):
        raise AssertionError("Benchmark returns do not match the recorded open prices")
    if not np.allclose(
        results["gross_return"].to_numpy(),
        results["position"].to_numpy() * results["benchmark_gross_return"].to_numpy(),
    ):
        raise AssertionError("Strategy returns are not aligned to benchmark intervals")

    expected_strategy_equity = INITIAL_CAPITAL * np.cumprod(
        1 + results["net_return"].to_numpy(dtype=float)
    )
    expected_benchmark_equity = INITIAL_CAPITAL * np.cumprod(
        1 + results["benchmark_net_return"].to_numpy(dtype=float)
    )
    if not np.allclose(results["strategy_equity"], expected_strategy_equity):
        raise AssertionError("Strategy equity does not match compounded net returns")
    if not np.allclose(results["benchmark_equity"], expected_benchmark_equity):
        raise AssertionError("Benchmark equity does not match compounded net returns")
    if not results["final_liquidation"].iloc[-1] or results["final_liquidation"].iloc[:-1].any():
        raise AssertionError("Final liquidation must be accounted for at the end")
    final_position = int(results["position"].iloc[-1])
    expected_liquidation_cost = (
        results["strategy_equity"].iloc[-1]
        * TRANSACTION_COST_RATE
        / (1 - TRANSACTION_COST_RATE)
        if final_position
        else 0.0
    )
    if not np.isclose(
        results["final_liquidation_cost"].iloc[-1], expected_liquidation_cost
    ):
        raise AssertionError("Final liquidation cost was not fully accounted for")


def save_equity_plot(results: pd.DataFrame) -> None:
    """Save strategy and buy-and-hold equity curves."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(results["execution_end_date"], results["strategy_equity"], label="Strategy")
    axis.plot(
        results["execution_end_date"],
        results["benchmark_equity"],
        label="Buy and hold",
    )
    axis.axhline(INITIAL_CAPITAL, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title("Out-of-sample strategy vs. buy-and-hold equity")
    axis.set_xlabel("Execution interval end date")
    axis.set_ylabel("Equity ($)")
    axis.legend()
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(PLOT_OUTPUT, dpi=150)
    plt.close(figure)


def write_summary(
    strategy_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    results: pd.DataFrame,
) -> None:
    """Write rules, timing assumptions, limitations, and realized metrics."""
    def metric_rows(metrics: dict[str, Any]) -> list[str]:
        fields = [
            ("Total return", "total_return", "{:.2%}"),
            ("Annualized return", "annualized_return", "{:.2%}"),
            ("Annualized volatility", "annualized_volatility", "{:.2%}"),
            ("Sharpe ratio", "sharpe_ratio", "{:.4f}"),
            ("Maximum drawdown", "maximum_drawdown", "{:.2%}"),
            ("Position changes", "position_changes", "{}"),
            ("Transaction costs", "transaction_costs", "${:,.2f}"),
        ]
        lines: list[str] = []
        for label, key, format_string in fields:
            value = metrics[key]
            rendered = "N/A" if value is None else format_string.format(value)
            lines.append(f"| {label} | {rendered} |")
        return lines

    start = results["signal_date"].min().date().isoformat()
    end = results["signal_date"].max().date().isoformat()
    summary = [
        "# SPY Strategy Backtest",
        "",
        "## Rules and execution",
        "",
        f"- Test signals only, from {start} through {end}; training and validation "
        "periods are excluded.",
        "- A positive predicted five-day return sets exposure to 1 (long SPY); "
        "a non-positive prediction sets exposure to 0 (cash).",
        "- Each prediction is assumed available at the signal date close. Exposure "
        "changes execute at the next trading day's open and apply to the following "
        "open-to-open return.",
        "- The benchmark holds SPY over the same executable intervals.",
        "- Initial capital is $10,000; there is no leverage, short selling, "
        "fractional execution constraint, or interest on cash.",
        "- Trading costs are 0.1% per unit of exposure change. Entry, position "
        "changes, and final liquidation are charged against portfolio equity.",
        "- Position-change counts include initial entry and final liquidation "
        "when those trades change exposure.",
        f"- Annualized metrics use {TRADING_DAYS_PER_YEAR} trading days; the "
        "Sharpe ratio assumes a zero risk-free rate.",
        "",
        "## Results",
        "",
        f"Executable intervals: {len(results)}",
        "",
        "| Metric | Strategy | Buy and hold |",
        "| --- | ---: | ---: |",
    ]
    strategy_lines = metric_rows(strategy_metrics)
    benchmark_lines = metric_rows(benchmark_metrics)
    summary.extend(
        f"| {row.split('|')[1].strip()} | {row.split('|')[2].strip()} | "
        f"{benchmark_row.split('|')[2].strip()} |"
        for row, benchmark_row in zip(strategy_lines, benchmark_lines)
    )
    summary.extend(
        [
            "",
            "## Limitations",
            "",
            "This is a simplified daily backtest. It assumes fills at recorded opens "
            "with no slippage, bid-ask spread, market impact, taxes, or partial fills. "
            "The fixed proportional cost is the only modeled execution friction. "
            "Signals use the prediction file's out-of-sample test period; this script "
            "does not retrain or tune the model. Returns are measured over consecutive "
            "trading-day open-to-open intervals, not the model's five-day target "
            "horizon. The first and last modeled execution boundaries therefore do "
            "not represent a continuous buy-and-hold position outside these intervals.",
            "",
        ]
    )
    SUMMARY_OUTPUT.write_text("\n".join(summary), encoding="utf-8")


def run_backtest() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build outputs and metrics without changing or retraining the model."""
    predictions = load_test_predictions()
    market = load_market_data()
    results = build_backtest(predictions, market)

    strategy_position_changes = int(results["position_change"].sum()) + int(
        results["position"].iloc[-1] == 1
    )
    benchmark_position_changes = 2
    strategy_metrics = calculate_metrics(
        results["net_return"],
        results["strategy_equity"],
        results["transaction_cost"],
        strategy_position_changes,
    )
    benchmark_metrics = calculate_metrics(
        results["benchmark_net_return"],
        results["benchmark_equity"],
        results["benchmark_transaction_cost"],
        benchmark_position_changes,
    )
    metrics = {
        "initial_capital": INITIAL_CAPITAL,
        "transaction_cost_rate": TRANSACTION_COST_RATE,
        "trading_days_per_year": TRADING_DAYS_PER_YEAR,
        "risk_free_rate": 0.0,
        "target_used_for_decisions": False,
        "return_interval": "next trading day's open to following trading day's open",
        "executable_intervals": len(results),
        "signal_date_range": {
            "start": results["signal_date"].min().date().isoformat(),
            "end": results["signal_date"].max().date().isoformat(),
        },
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_OUTPUT, index=False, date_format="%Y-%m-%d")
    METRICS_OUTPUT.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_equity_plot(results)
    write_summary(strategy_metrics, benchmark_metrics, results)
    return results, metrics


def main() -> None:
    """Run the validated out-of-sample backtest and save its reports."""
    try:
        results, metrics = run_backtest()
    except (OSError, ValueError, AssertionError) as exc:
        raise SystemExit(f"Backtest failed: {exc}") from exc

    print("BACKTEST VALIDATION PASSED")
    print(f"EXECUTABLE INTERVALS: {len(results)}")
    print(f"STRATEGY TOTAL RETURN: {metrics['strategy']['total_return']:.2%}")
    print(f"BUY-AND-HOLD TOTAL RETURN: {metrics['benchmark']['total_return']:.2%}")
    print(f"STRATEGY TRANSACTION COSTS: ${metrics['strategy']['transaction_costs']:,.2f}")
    print("BACKTEST COMPLETE")


if __name__ == "__main__":
    main()
