# GitHub Actions automation

The **SPY Market Analytics Pipeline** workflow runs the project modules in
dependency order:

1. `python -m src.data_ingestion`
2. `python -m src.features`
3. `python -m src.eda`
4. `python -m src.train_model`
5. `python -m src.backtest`

It runs on weekdays at 23:00 UTC and can also be started manually. Each run
uses Python 3.12, installs `requirements.txt`, and has a 30-minute timeout.
Reports are uploaded as the `spy-market-analytics-reports` artifact for 14 days,
even if an earlier pipeline step fails. The workflow does not commit or push
generated data, models, or reports; no broker or live-trading integration is
configured.

## Run manually

From the repository root, run the five commands above in order. With the
project's virtual environment, for example:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data_ingestion
python -m src.features
python -m src.eda
python -m src.train_model
python -m src.backtest
```

On GitHub, open **Actions**, select **SPY Market Analytics Pipeline**, choose
**Run workflow**, and confirm the branch. Open the completed run to inspect
step logs or download its reports artifact.

## Troubleshooting

- **External data request fails:** Ingestion depends on Yahoo Finance and FRED.
  Their external APIs may be temporarily unavailable or may fail because of
  network problems, service changes, or rate limits. Inspect the ingestion
  step logs and retry the workflow later; failures are not silently replaced
  with stale data.
- **Dependency installation fails:** Check the `requirements.txt` pins and the
  Python 3.12 setup step, then inspect the install log for package or wheel
  compatibility errors.
- **A later stage fails:** The workflow stops at the failing stage. Use that
  step's logs to identify missing or invalid input artifacts. The report
  artifact upload is attempted regardless of pipeline success.
- **No reports artifact is available:** Check whether the upload step ran and
  whether `reports/` contained files; the upload step warns when no files are
  found.
