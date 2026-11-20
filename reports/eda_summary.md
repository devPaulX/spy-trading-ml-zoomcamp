# Exploratory Data Analysis

## Target

`future_5d_return` is the SPY close price five trading days ahead divided by the current close, minus one. The final five observations have no known target yet and are excluded from target statistics and correlations.

### Summary statistics

| Statistic | Value |
| --- | ---: |
| count | 2965.00000000 |
| mean | 0.00272723 |
| std | 0.02262647 |
| min | -0.17969281 |
| 25% | -0.00712149 |
| 50% | 0.00430426 |
| 75% | 0.01463561 |
| max | 0.17358122 |

Observed targets: 2965
Missing targets: 5

## Feature definitions

All rolling windows use current and prior observations only. `yield_lag_1d` uses the previous trading day's yield.

| Feature | Definition |
| --- | --- |
| `ret_1d` | SPY close percentage return over the previous trading day. |
| `ret_5d` | SPY close percentage return over the previous five trading days. |
| `ret_20d` | SPY close percentage return over the previous twenty trading days. |
| `ma_5_ratio` | Current SPY close divided by its trailing five-day average close, minus one. |
| `ma_20_ratio` | Current SPY close divided by its trailing twenty-day average close, minus one. |
| `volatility_20d` | Trailing twenty-day sample standard deviation of one-day SPY returns. |
| `volume_ratio_20d` | Current SPY volume divided by its trailing twenty-day average volume. |
| `yield_lag_1d` | US 10-year Treasury yield from the previous trading day. |
| `yield_change_5d` | Five-trading-day change in the one-day-lagged US 10-year yield. |
| `weekday_Monday` | Indicator that the observation date is a Monday. |
| `weekday_Tuesday` | Indicator that the observation date is a Tuesday. |
| `weekday_Wednesday` | Indicator that the observation date is a Wednesday. |
| `weekday_Thursday` | Indicator that the observation date is a Thursday. |
| `weekday_Friday` | Indicator that the observation date is a Friday. |

## Pearson correlations with target

Correlations use rows with observed targets.

| Feature | Correlation |
| --- | ---: |
| `ret_1d` | -0.05063328 |
| `ret_5d` | -0.05312747 |
| `ret_20d` | -0.04144862 |
| `ma_5_ratio` | -0.07207136 |
| `ma_20_ratio` | -0.04724293 |
| `volatility_20d` | 0.09661598 |
| `volume_ratio_20d` | -0.01012432 |
| `yield_lag_1d` | -0.01818953 |
| `yield_change_5d` | -0.01589175 |
| `weekday_Monday` | -0.00008711 |
| `weekday_Tuesday` | 0.00141483 |
| `weekday_Wednesday` | 0.00227151 |
| `weekday_Thursday` | -0.00081631 |
| `weekday_Friday` | -0.00281963 |
