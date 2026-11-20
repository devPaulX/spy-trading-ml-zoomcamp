# SPY Strategy Backtest

## Rules and execution

- Test signals only, from 2025-02-07 through 2026-11-13; training and validation periods are excluded.
- A positive predicted five-day return sets exposure to 1 (long SPY); a non-positive prediction sets exposure to 0 (cash).
- Each prediction is assumed available at the signal date close. Exposure changes execute at the next trading day's open and apply to the following open-to-open return.
- The benchmark holds SPY over the same executable intervals.
- Initial capital is $10,000; there is no leverage, short selling, fractional execution constraint, or interest on cash.
- Trading costs are 0.1% per unit of exposure change. Entry, position changes, and final liquidation are charged against portfolio equity.
- Position-change counts include initial entry and final liquidation when those trades change exposure.
- Annualized metrics use 252 trading days; the Sharpe ratio assumes a zero risk-free rate.

## Results

Executable intervals: 445

| Metric | Strategy | Buy and hold |
| --- | ---: | ---: |
| Total return | 11.04% | 24.40% |
| Annualized return | 6.11% | 13.16% |
| Annualized volatility | 16.27% | 17.38% |
| Sharpe ratio | 0.4460 | 0.7978 |
| Maximum drawdown | -24.29% | -19.77% |
| Position changes | 18 | 2 |
| Transaction costs | $170.85 | $22.45 |

## Limitations

This is a simplified daily backtest. It assumes fills at recorded opens with no slippage, bid-ask spread, market impact, taxes, or partial fills. The fixed proportional cost is the only modeled execution friction. Signals use the prediction file's out-of-sample test period; this script does not retrain or tune the model. Returns are measured over consecutive trading-day open-to-open intervals, not the model's five-day target horizon. The first and last modeled execution boundaries therefore do not represent a continuous buy-and-hold position outside these intervals.
