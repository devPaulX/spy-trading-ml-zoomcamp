# Reproducibility guide

Run commands from the repository root. The full pipeline downloads fresh
market data, so results can change as providers revise data or add observations.
The source code does not require brokerage credentials and does not place trades.

## Environment prerequisites

- Python 3.12
- Git
- Network access to Yahoo Finance and FRED for data ingestion
- A virtual environment is recommended

The GitHub Actions workflow also uses Python 3.12. Dependencies are pinned in
[`requirements.txt`](../requirements.txt).

## Install

On Linux or macOS:

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, create and activate the environment with:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Retrieve data

From the repository root, run:

```bash
python -m src.data_ingestion
```

The ingestion module downloads adjusted daily SPY OHLCV from Yahoo Finance
starting 2015-01-01 and the FRED `DGS10` series, joins them on date, forward
fills missing yields, and drops rows without a yield. It saves
`data/market_data.csv`. No API key is required. The `data/` directory is
ignored by Git, so the input data is not committed.

If either provider is unavailable, inspect the command error and retry after
the provider/network issue is resolved. The pipeline intentionally does not
substitute stale or fabricated data.

## Execute the pipeline

Run all five modules in this exact order:

```bash
python -m src.data_ingestion
python -m src.features
python -m src.eda
python -m src.train_model
python -m src.backtest
```

Each command reads and writes relative to the project root. Later stages depend
on the preceding stage's outputs.

## Expected outputs

| Stage | Outputs |
| --- | --- |
| Ingestion | `data/market_data.csv` |
| Features | `data/features.csv` |
| EDA | `reports/eda_summary.md`, `reports/feature_correlations.csv`, `reports/eda.png` |
| Model training | `models/random_forest.joblib`, `models/random_forest_metadata.json`, `reports/test_predictions.csv`, `reports/model_metrics.json`, `reports/feature_importance.csv`, `reports/model_evaluation.png` |
| Backtest | `reports/backtest_results.csv`, `reports/backtest_metrics.json`, `reports/backtest_equity.png`, `reports/backtest_summary.md` |

Data and model directories are ignored by Git. Reports are written under
`reports/`; the files in the repository are report snapshots and are
overwritten by a successful run.

## Validation checks

- Ingestion checks that Yahoo Finance returned data, that the dataset contains
  more than 1,000 rows, and that SPY close values are present.
- Feature generation sorts by date, computes only backward-looking inputs,
  drops rows without feature history, and retains the final five rows with
  missing target values.
- Model training requires the expected feature/target columns, finite model
  inputs, non-overlapping chronological partitions, and purges five rows at
  both train/validation boundaries. The test period is held out from fitting
  and is used only for evaluation and test predictions.
- The backtest verifies that predictions match the recorded test period,
  signals are unique and chronological, execution follows signals, returns use
  matching open-to-open intervals, exposure is only 0 or 1, returns are finite,
  costs are nonnegative, final liquidation is charged, and equity matches
  compounded net returns. Successful execution prints
  `BACKTEST VALIDATION PASSED` and `BACKTEST COMPLETE`.

Expected outputs and metrics may vary when data is refreshed. To see the
specific snapshot summarized in the README, inspect
[`reports/model_metrics.json`](../reports/model_metrics.json) and
[`reports/backtest_metrics.json`](../reports/backtest_metrics.json).

## Common failures and remedies

| Symptom | Likely cause | Remedy |
| --- | --- | --- |
| `No such file or directory` for data or reports | A module was run outside the repository root, or an earlier stage has not run | Change to the repository root and run the pipeline stages in order |
| Yahoo Finance or FRED request error | Network/provider outage, rate limit, or upstream format change | Check connectivity and provider availability, inspect the full error, and retry later |
| `pip install` fails | Unsupported Python version or dependency installation problem | Confirm Python 3.12 is active; use a fresh `.venv` and inspect the failing package's pip output |
| Missing columns or invalid values | Upstream schema/data change or incomplete input file | Re-run ingestion if appropriate; inspect `data/market_data.csv` headers and values before continuing |
| Backtest predictions do not match test period | Predictions or model metrics came from different pipeline runs | Run `src.train_model` successfully after ingestion/features, then run `src.backtest` without mixing report snapshots |
| No reports are available from GitHub Actions | Workflow stopped before report generation or artifact upload had no files | Inspect the failing job step; report upload is attempted even after failure |

## GitHub Actions

The [`market_pipeline.yml`](../.github/workflows/market_pipeline.yml) workflow
uses Python 3.12, installs the pinned requirements, and runs all pipeline
stages. It is scheduled on weekdays at 23:00 UTC and can also be started in
GitHub under **Actions → SPY Market Analytics Pipeline → Run workflow**.
Reports are uploaded as `spy-market-analytics-reports` for 14 days. The
workflow does not commit or push generated data, models, or reports.
